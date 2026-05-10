"""
图像预处理模块
职责：
  - 批次侦察与文件排序
  - ROI 标定与 UI 掩膜生成
  - 单帧数据加载（纯净 / 脏数据双轨制）
  - 全局归一化与时空图构建
  - 数据持久化（npy 文件输出）
  - 数据对抗增强（可选工具）

注意：本模块不包含任何可视化功能，所有绘图任务请使用 visualization.py。
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =====================================================================
# 第一部分：批次侦察与控制流辅助
# =====================================================================

def analyze_batch_context(input_folder):
    """
    功能: 批次侦察兵。扫描输入文件夹，判断这是纯净训练批次还是实拍脏数据批次。
    返回: 字典 context，供主程序进行路由分发。
    """
    folder_path = Path(input_folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    all_files = list(folder_path.glob("*.*"))
    image_files = [f for f in all_files if f.suffix.lower() in ['.img', '.png', '.jpg']]

    if not image_files:
        raise ValueError(f"在 {folder_path} 中没有找到任何图片文件！")

    first_image = image_files[0]
    ext = first_image.suffix.lower()

    context = {
        "first_frame_path": first_image,
        "image_list": image_files,
        "total_count": len(image_files)
    }

    if ext == '.img':
        context["batch_type"] = "clean"
        print(f"🔍 侦察完毕：检测到 {context['total_count']} 张 .img 纯净数据。")

    elif ext in ['.png', '.jpg']:
        context["batch_type"] = "dirty"
        template_path = folder_path / "ui_mask_template.png"
        if not template_path.exists():
            raise FileNotFoundError(
                f"\n🚨 致命错误：这是实拍批次，但在文件夹中找不到 'ui_mask_template.png'！\n"
                f"请放入全黑的UI掩膜底片后再运行程序。"
            )
        context["template_path"] = template_path
        print(f"🔍 侦察完毕：检测到 {context['total_count']} 张实拍脏数据，且 UI 底片已就绪。")

    return context

# =====================================================================
# 第二部分：标定与掩膜生成
# =====================================================================

def get_roi_from_first_frame(image_path):
    """
    功能: 弹出一个 GUI 窗口，让用户手工框选核心的衍射条纹区域。
    注意: 若图像为 16-bit,将临时映射到 8-bit 以便显示，但不影响原数据。
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 16-bit 图像临时映射到 8-bit（仅用于显示）
    if img.dtype == np.uint16:
        img_display = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_display = img

    print("\n" + "=" * 40 + "\n📢 标定模式启动\n请框选核心衍射条纹并按【回车键】确认。\n" + "=" * 40 + "\n")
    roi_rect = cv2.selectROI("Select RHEED ROI", img_display, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi_rect[2] == 0 or roi_rect[3] == 0:
        raise ValueError("标定取消或未画框！程序中断。")
    return roi_rect

def generate_ui_mask(template_path, roi_rect, target_resolution=(256, 256)):
    """
    功能: 读取全黑底片，提取像素级的绝对掩膜 (UI 污染区)。
    规定: 缩放时必须使用最近邻插值，以保证掩膜边缘仍是锐利的布尔值。
    """
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    x, y, w, h = roi_rect
    cropped_template = template[y:y + h, x:x + w]
    # 最近邻插值，确保 UI 线条一刀切
    resized_template = cv2.resize(cropped_template, target_resolution, interpolation=cv2.INTER_NEAREST)
    ui_mask = resized_template > 10
    return ui_mask

# =====================================================================
# 第三部分：归一化与单帧加载器（双轨制）
# =====================================================================

def normalize_physics_matrix(matrix, global_min=None, global_max=None):
    """
    功能: 将矩阵映射到 0.0 ~ 1.0。
    参数: 
        global_min, global_max : 如果提供，则执行基于全局范围的归一化；
                                否则对矩阵进行独立归一化。
    说明: 使用 np.nanmin / np.nanmax 安全处理 NaN 像素。
    """
    if global_min is None or global_max is None:
        min_val = np.nanmin(matrix)
        max_val = np.nanmax(matrix)
    else:
        min_val, max_val = global_min, global_max

    if max_val - min_val == 0:
        return matrix - min_val  # 全零矩阵，避免除零
    return (matrix - min_val) / (max_val - min_val)

def process_clean_img_file(image_path, roi_rect, target_resolution=(256, 256)):
    """
    轨道 A:纯净数据流。
    处理前期的高位深纯净训练数据 (.img)，输出未归一化的 float32 矩阵。
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    x, y, w, h = roi_rect
    cropped_img = img[y:y + h, x:x + w]

    if len(cropped_img.shape) == 3:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    float_matrix = cropped_img.astype(np.float32)
    # 物理插值（缩小）保留积分强度
    resized_matrix = cv2.resize(float_matrix, target_resolution, interpolation=cv2.INTER_AREA)
    return resized_matrix  # 归一化延迟至全局处理

def process_dirty_png_file(image_path, roi_rect, ui_mask, target_resolution=(256, 256)):
    """
    轨道 B:实拍数据流。
    处理后期的实拍实验截图 (.png)，输出未归一化的 float32 矩阵,UI 区域已置为 NaN。
    """
    img = cv2.imread(str(image_path))
    x, y, w, h = roi_rect
    cropped_img = img[y:y + h, x:x + w]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    float_matrix = gray_img.astype(np.float32)
    resized_matrix = cv2.resize(float_matrix, target_resolution, interpolation=cv2.INTER_AREA)

    # 物理屏蔽：将 UI 污染像素彻底移除
    resized_matrix[ui_mask] = np.nan
    return resized_matrix  # 归一化延迟至全局处理

# =====================================================================
# 第四部分：实验批次全局处理流水线
# =====================================================================

def process_experiment_folder(input_folder, target_resolution=(256, 256)):
    """
    实验批次主处理函数。
    流程：
      1. 侦察批次类型（当前仅支持 dirty 批次）。
      2. 打开第一帧进行 ROI 标定，生成 UI 掩膜。
      3. 按文件名数字排序加载所有帧，执行裁剪、缩放、NaN 屏蔽。
      4. 计算全局强度极值，对全部帧执行全局归一化。
      5. 构建时空图立方体 (T, H, W) 及有效性掩膜。
      6. 保存立方体与掩膜至 processed/ 子目录。
    返回:
        spacetime_cube : np.ndarray, 形状 (T, 256, 256)，已归一化
        validity_mask  : np.ndarray, 形状 (T, 256, 256)，True 表示有效像素
    """
    # ---- 1. 侦察 ----
    context = analyze_batch_context(input_folder)
    if context["batch_type"] != "dirty":
        raise NotImplementedError("当前实验批次处理仅支持 dirty (PNG) 模式。")

    # ---- 2. 标定与掩膜 ----
    roi_rect = get_roi_from_first_frame(context["first_frame_path"])
    ui_mask = generate_ui_mask(context["template_path"], roi_rect, target_resolution)

    # ---- 3. 帧排序 ----
    png_files = [f for f in context["image_list"] if f.suffix.lower() == '.png']
    # 按数字文件名排序（支持非 1 开始的序列）
    try:
        png_files_sorted = sorted(png_files, key=lambda f: int(f.stem))
    except ValueError:
        png_files_sorted = sorted(png_files, key=lambda f: f.stem)
        print("⚠️ 警告：文件名不完全为数字，已按字符串顺序排列，请确认时间顺序无误。")

    if not png_files_sorted:
        raise ValueError("未发现任何 PNG 文件。")

    print(f"📂 开始处理 {len(png_files_sorted)} 帧...")
    frames = []
    for file_path in tqdm(png_files_sorted, desc="帧预处理"):
        frame = process_dirty_png_file(file_path, roi_rect, ui_mask, target_resolution)
        frames.append(frame)

    # ---- 4. 全局归一化 ----
    stacked = np.stack(frames, axis=0)  # (T, H, W)
    global_min = np.nanmin(stacked)
    global_max = np.nanmax(stacked)
    print(f"📊 全局强度范围: [{global_min:.3f}, {global_max:.3f}]")

    normalized_frames = [normalize_physics_matrix(f, global_min, global_max) for f in frames]
    spacetime_cube = np.stack(normalized_frames, axis=0)  # 时空立方体

    # 有效性掩膜：与 NaN 相反
    validity_mask = ~np.isnan(spacetime_cube)

    # ---- 5. 持久化 ----
    output_dir = Path(input_folder) / "processed"
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "spacetime_cube.npy", spacetime_cube)
    np.save(output_dir / "validity_mask.npy", validity_mask)
    print(f"💾 时空立方体已保存至 {output_dir / 'spacetime_cube.npy'}")

    return spacetime_cube, validity_mask

# =====================================================================
# 第五部分：数据对抗增强（工具函数）
# =====================================================================

def simulate_ui_degradation(clean_matrix):
    """
    功能: 对纯净数据反向污染。在完美矩阵上人为注入 NaN 破坏点，
         用于训练下游算法的鲁棒性。
    """
    degraded_matrix = clean_matrix.copy()
    h, w = degraded_matrix.shape

    # 中心区域十字准星污染（宽 3 像素）
    center_y, center_x = h // 2, w // 2
    degraded_matrix[center_y - 1:center_y + 2, :] = np.nan
    degraded_matrix[:, center_x - 1:center_x + 2] = np.nan

    return degraded_matrix