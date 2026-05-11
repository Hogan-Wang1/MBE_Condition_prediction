"""
预处理模块（流式精简版）
职责：
  - 批次侦察（.img 或 .png 单一格式）
  - 固定比例多 ROI 映射（与窗口缩放解耦）
  - 双轨单帧加载（纯净 / 脏 UI 屏蔽）
  - 前一帧漂移追踪校正（失败不更新模板）
  - 流式生成器（内存安全，一次解码多 ROI 并行）
  - memmap 持久化辅助

注意：不包含背景扣除，适合以相对强度/位置为特征的分析。
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Generator, Any

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# 硬编码固定比例 ROI（物理几何，相对截图中心）
# 比例推导见系统提示词，不可随意修改
# =====================================================================
LEFT_RATIO   = -25 / 63   # ≈ -0.3968
RIGHT_RATIO  =  25 / 63   # ≈  0.3968
TOP_RATIO    =   2 / 19   # ≈  0.1053
BOTTOM_RATIO =  -6 / 19   # ≈ -0.3158

def ratio_roi_to_pixels(image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """将硬编码比例转换为当前帧的像素矩形 (x, y, w, h)。"""
    H, W = image_shape
    cx, cy = W / 2.0, H / 2.0
    x_left  = cx + LEFT_RATIO * W
    x_right = cx + RIGHT_RATIO * W
    y_top   = cy - TOP_RATIO * H      # 像素 Y 向下，故减
    y_bottom= cy - BOTTOM_RATIO * H   # BOTTOM_RATIO 为负，减负得加
    x = int(round(x_left))
    y = int(round(y_top))
    w = int(round(x_right - x_left))
    h = int(round(y_bottom - y_top))
    # 边界保护
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = min(w, W - x)
    h = min(h, H - y)
    return x, y, w, h

# =====================================================================
# 批次侦察（确保单一格式）
# =====================================================================
def analyze_batch_context(input_folder: str) -> Dict[str, Any]:
    """扫描文件夹，返回 batch_type ('clean' 或 'dirty') 及文件列表。"""
    folder = Path(input_folder)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    files = list(folder.glob("*.*"))
    img_files = [f for f in files if f.suffix.lower() in ['.img', '.png', '.jpg']]
    if not img_files:
        raise ValueError(f"未在 {folder} 中发现任何图像文件。")

    # 确保单一格式
    exts = set(f.suffix.lower() for f in img_files)
    if len(exts) > 1:
        raise ValueError(f"检测到混合格式 {exts}，当前要求单一批次仅一种格式。")

    ctx = {
        "first_frame": img_files[0],
        "image_list": img_files,
        "total_count": len(img_files),
        "batch_type": "clean" if img_files[0].suffix.lower() == '.img' else "dirty"
    }
    if ctx["batch_type"] == "dirty":
        template = folder / "ui_mask_template.png"
        if not template.exists():
            raise FileNotFoundError("脏数据批次缺少 ui_mask_template.png。")
        ctx["template_path"] = template

    logger.info(f"侦察完毕：{ctx['total_count']} 帧 {ctx['batch_type']} 数据。")
    return ctx

def sort_image_files(file_list: List[Path]) -> List[Path]:
    """按数字文件名排序，失败则按字符串顺序并警告。"""
    try:
        return sorted(file_list, key=lambda f: int(f.stem))
    except ValueError:
        logger.warning("文件名非纯数字，已按字符串排序，请确认时间顺序。")
        return sorted(file_list, key=lambda f: f.stem)

# =====================================================================
# 单帧加载器（双轨，无背景扣除）
# =====================================================================
def read_img_clean(file_path: Path) -> np.ndarray:
    """读取 .img 文件，返回 float32 全图。"""
    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"无法读取 {file_path}")
    if img.dtype == np.uint16:
        img = img.astype(np.float32)
    else:
        img = img.astype(np.float32)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def read_png_dirty(file_path: Path, ui_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """读取 .png 截图，返回 float32 全图，UI 区域置 NaN。"""
    img = cv2.imread(str(file_path))
    if img is None:
        raise IOError(f"无法读取 {file_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if ui_mask is not None:
        # 若掩膜尺寸变化则用最近邻缩放以保持锐利布尔
        if ui_mask.shape != gray.shape:
            ui_mask = cv2.resize(ui_mask.astype(np.uint8),
                                 (gray.shape[1], gray.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        gray[ui_mask.astype(bool)] = np.nan
    return gray

def generate_full_ui_mask(template_path: Path) -> np.ndarray:
    """从全黑底片生成全图 UI 污染掩膜（True=污染）。"""
    tmpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise IOError(f"无法读取模板 {template_path}")
    mask = tmpl > 10
    logger.info(f"全图 UI 掩膜已加载，{np.sum(mask)} 像素被屏蔽。")
    return mask

# =====================================================================
# 前一帧漂移追踪器
# =====================================================================
class DriftTracker:
    """利用前一帧模板匹配实现束流漂移补偿，失败不更新模板。"""
    def __init__(self,
                 search_margin: int = 15,
                 corr_threshold: float = 0.3,
                 black_threshold: float = 1.0):
        self.search_margin = search_margin
        self.corr_threshold = corr_threshold
        self.black_threshold = black_threshold
        self.template = None          # 前一帧 ROI（原生尺寸）
        self.template_shape = None

    def initialize(self, roi_image: np.ndarray):
        """用第一帧的 ROI 子图初始化模板。"""
        self.template = roi_image.copy()
        self.template_shape = roi_image.shape[:2]
        logger.debug(f"DriftTracker 初始化，模板尺寸 {self.template_shape}")

    def _sanitize(self, img: np.ndarray) -> np.ndarray:
        """替换 NaN 为 0 供 OpenCV 处理。"""
        out = img.copy()
        out[np.isnan(out)] = 0.0
        return out

    def track(self, full_frame: np.ndarray,
              nominal_rect: Tuple[int, int, int, int]
              ) -> Tuple[Tuple[int, int, int, int], Tuple[float, float], bool]:
        """
        在 full_frame 中搜索模板，返回校正矩形、漂移向量、可靠性。
        若不可靠，不更新内部模板（由调用方根据返回值决定）。
        """
        x, y, w, h = nominal_rect
        H, W = full_frame.shape[:2]

        if self.template is None:
            return nominal_rect, (0.0, 0.0), False

        # 模板过暗（束斑消失）→ 追踪失败，不更新模板
        if np.nanmean(self.template) < self.black_threshold:
            return nominal_rect, (0.0, 0.0), False

        # 搜索区域
        sx1 = max(0, x - self.search_margin)
        sy1 = max(0, y - self.search_margin)
        sx2 = min(W, x + w + self.search_margin)
        sy2 = min(H, y + h + self.search_margin)
        if sx1 >= sx2 or sy1 >= sy2:
            return nominal_rect, (0.0, 0.0), False

        search_region = self._sanitize(full_frame[sy1:sy2, sx1:sx2])
        tmpl = self._sanitize(self.template)

        # 模板匹配
        res = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < self.corr_threshold:
            return nominal_rect, (0.0, 0.0), False

        dx = max_loc[0] - self.search_margin
        dy = max_loc[1] - self.search_margin

        corrected_x = int(x + dx)
        corrected_y = int(y + dy)
        corrected_x = max(0, min(corrected_x, W - w))
        corrected_y = max(0, min(corrected_y, H - h))
        corrected_rect = (corrected_x, corrected_y, w, h)

        # 注意：不在内部更新模板，由调用方在确认可靠后显式更新
        return corrected_rect, (dx, dy), True

# =====================================================================
# 流式生成器（核心）
# =====================================================================
def stream_rheed_frames(
    input_folder: str,
    roi_ratios: Optional[List[Dict[str, float]]] = None,
    target_size: Tuple[int, int] = (128, 128),
    enable_drift: bool = True,
    drift_search_margin: int = 15,
    drift_corr_threshold: float = 0.3,
    drift_black_threshold: float = 1.0
) -> Generator[Dict[str, Any], None, None]:
    """
    流式读取 RHEED 图像序列，提取多 ROI，前一帧漂移校正。
    
    参数：
        input_folder : 文件夹路径，仅含一种格式（.img 或 .png）。
        roi_ratios   : ROI 比例定义列表，默认使用硬编码单 RO。
        target_size  : 输出 ROI 的尺寸 (宽, 高)。
        enable_drift : 是否启用漂移校正。
        drift_search_margin : 搜索扩展像素。
        drift_corr_threshold: 相关系数阈值。
        drift_black_threshold: 模板最暗平均强度。
    
    生成器 yield 字典：
        {
            'frame_id': int,
            'rois': {'roi_name': np.ndarray(目标尺寸), ...},
            'drift': (dx, dy),
            'drift_reliable': bool,
            'metadata': {...}
        }
    """
    ctx = analyze_batch_context(input_folder)
    is_clean = ctx["batch_type"] == "clean"
    image_files = sort_image_files(ctx["image_list"])

    # 加载 UI 掩膜（仅脏数据）
    ui_mask = None
    if not is_clean:
        ui_mask = generate_full_ui_mask(ctx["template_path"])

    # 默认 ROI：硬编码固定框
    if roi_ratios is None:
        roi_ratios = [{"name": "main_spot",
                       "left": LEFT_RATIO, "right": RIGHT_RATIO,
                       "top": TOP_RATIO, "bottom": BOTTOM_RATIO}]

    # 初始化漂移追踪器
    tracker = None
    if enable_drift:
        tracker = DriftTracker(search_margin=drift_search_margin,
                               corr_threshold=drift_corr_threshold,
                               black_threshold=drift_black_threshold)

    for idx, file_path in enumerate(image_files):
        # ---- 1. 加载全图 ----
        if is_clean:
            full = read_img_clean(file_path)
        else:
            full = read_png_dirty(file_path, ui_mask)

        H, W = full.shape[:2]

        # ---- 2. 计算所有 ROI 的名义矩形 ----
        nominal_rects = []
        for roi_def in roi_ratios:
            if all(k in roi_def for k in ("left","right","top","bottom")):
                left  = int(W/2 + roi_def["left"] * W)
                right = int(W/2 + roi_def["right"] * W)
                top   = int(H/2 - roi_def["top"] * H)
                bottom= int(H/2 - roi_def["bottom"] * H)
                x = min(left, right); y = min(top, bottom)
                w = abs(right - left); h = abs(bottom - top)
                x = max(0, min(x, W-1))
                y = max(0, min(y, H-1))
                w = min(w, W - x)
                h = min(h, H - y)
                rect = (x, y, w, h)
            else:
                raise ValueError(f"ROI 定义缺少键: {roi_def}")
            nominal_rects.append(rect)

        # ---- 3. 漂移校正 ----
        primary_rect = nominal_rects[0]   # 所有 ROI 共用同一漂移
        drift_vec = (0.0, 0.0)
        drift_ok = False

        if enable_drift and tracker is not None:
            if idx == 0:
                # 第一帧初始化模板（原生尺寸）
                x0, y0, w0, h0 = primary_rect
                init_roi = full[y0:y0+h0, x0:x0+w0].copy()
                tracker.initialize(init_roi)
                corrected_rects = nominal_rects
            else:
                corr_primary, drift_vec, drift_ok = tracker.track(full, primary_rect)
                dx = corr_primary[0] - primary_rect[0]
                dy = corr_primary[1] - primary_rect[1]
                corrected_rects = []
                for rect in nominal_rects:
                    nx = rect[0] + dx
                    ny = rect[1] + dy
                    nx = max(0, min(nx, W - rect[2]))
                    ny = max(0, min(ny, H - rect[3]))
                    corrected_rects.append((nx, ny, rect[2], rect[3]))
        else:
            corrected_rects = nominal_rects

        # ---- 4. 提取 ROI 并缩放到目标尺寸 ----
        roi_dict = {}
        for (x, y, w, h), roi_def in zip(corrected_rects, roi_ratios):
            sub = full[y:y+h, x:x+w].copy()
            # 缩放处理 NaN
            nan_mask = np.isnan(sub)
            sub_temp = np.where(nan_mask, 0.0, sub)
            if sub.shape[0] != target_size[1] or sub.shape[1] != target_size[0]:
                sub_resized = cv2.resize(sub_temp, target_size,
                                         interpolation=cv2.INTER_AREA)
                if np.any(nan_mask):
                    nan_mask_u8 = nan_mask.astype(np.uint8) * 255
                    nan_mask_rz = cv2.resize(nan_mask_u8, target_size,
                                             interpolation=cv2.INTER_NEAREST) > 127
                    sub_resized[nan_mask_rz] = np.nan
            else:
                sub_resized = sub_temp
                if np.any(nan_mask):
                    sub_resized[nan_mask] = np.nan
            roi_dict[roi_def["name"]] = sub_resized.astype(np.float32)

        # ---- 5. 若漂移成功，更新模板（原生尺寸，未缩放） ----
        if enable_drift and tracker is not None and idx > 0 and drift_ok:
            px, py, pw, ph = corrected_rects[0]
            new_tmpl = full[py:py+ph, px:px+pw].copy()
            tracker.template = new_tmpl   # 关键：仅成功时更新

        # ---- 6. 组装输出 ----
        yield {
            'frame_id': idx,
            'rois': roi_dict,
            'drift': drift_vec,
            'drift_reliable': drift_ok,
            'metadata': {
                'file_name': file_path.name,
                'original_shape': (H, W),
                'primary_nominal_rect': primary_rect
            }
        }

# =====================================================================
# memmap 持久化辅助
# =====================================================================
def process_to_memmap(input_folder: str,
                      output_dir: str,
                      roi_ratios: Optional[List[Dict[str, float]]] = None,
                      target_size: Tuple[int, int] = (128, 128),
                      enable_drift: bool = True,
                      **drift_kwargs) -> None:
    """将流式输出写入内存映射文件，每个 ROI 一个 .dat。"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 第一趟统计
    roi_names = None
    total = 0
    for data in stream_rheed_frames(input_folder, roi_ratios, target_size,
                                    enable_drift, **drift_kwargs):
        if roi_names is None:
            roi_names = list(data['rois'].keys())
        total += 1
    if total == 0:
        raise RuntimeError("无有效帧。")

    H, W = target_size[1], target_size[0]
    memmaps = {}
    for name in roi_names:
        mm = np.memmap(str(out_path / f"{name}.dat"), dtype='float32',
                       mode='w+', shape=(total, H, W))
        memmaps[name] = mm

    # 第二趟填充
    idx = 0
    for data in stream_rheed_frames(input_folder, roi_ratios, target_size,
                                    enable_drift, **drift_kwargs):
        for name, sub in data['rois'].items():
            memmaps[name][idx] = sub
        idx += 1
        if idx % 100 == 0:
            logger.info(f"写入进度 {idx}/{total}")

    for name, mm in memmaps.items():
        mm.flush()
        np.save(out_path / f"{name}_shape.npy", np.array([total, H, W]))
    logger.info("memmap 写入完毕。")