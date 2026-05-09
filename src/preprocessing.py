import cv2
import numpy as np
from pathlib import Path

# =====================================================================
# 第一部分：批次侦察与控制流辅助 (Smart Router & Context Awareness)
# 对应策略：双轨制加载的“大脑”，自动识别数据类型
# =====================================================================

def analyze_batch_context(input_folder):
    """
    功能: 批次侦察兵。扫描输入文件夹，判断这是纯净训练批次还是实拍脏数据批次。
    返回: 字典 context,供主程序(batch_processor.py)进行路由分发。
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
# 第二部分：标定与掩膜生成 (Calibration & Masking)
# =====================================================================

def get_roi_from_first_frame(image_path):
    """
    功能: 弹出一个 GUI 窗口，让用户手工框选核心的衍射条纹区域。
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 【工程修复】如果是 16-bit (0-65535) 的训练数据，直接显示会全黑。
    # 必须将其在显示内存中临时映射到 8-bit，但不会影响实际数据。
    if img.dtype == np.uint16:
        img_display = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_display = img

    print("\n" + "="*40 + "\n📢 标定模式启动\n请框选核心衍射条纹并按【回车键】确认。\n" + "="*40 + "\n")
    roi_rect = cv2.selectROI("Select RHEED ROI", img_display, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    if roi_rect[2] == 0 or roi_rect[3] == 0: 
        raise ValueError("标定取消或未画框！程序中断。")
    return roi_rect

def generate_ui_mask(template_path, roi_rect, target_resolution=(256, 256)):
    """
    功能: 读取全黑底片，提取像素级的绝对掩膜 (UI 污染区)。
    """
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    x, y, w, h = roi_rect
    cropped_template = template[y:y+h, x:x+w]
    # 【核心】必须用最近邻插值 (INTER_NEAREST)，保证UI线条依然是一刀切的锐利边界
    resized_template = cv2.resize(cropped_template, target_resolution, interpolation=cv2.INTER_NEAREST)
    ui_mask = resized_template > 10
    return ui_mask

# =====================================================================
# 第三部分：核心数学归一化与双轨加载器 (Dual-Loader & Normalization)
# 对应策略：彻底抹平图像深度的物理差异，为后续的相对几何计算铺平道路
# =====================================================================

def normalize_physics_matrix(matrix):
    """
    功能: 【归一化处理策略】将任意深度的矩阵，强行映射到 0.0 ~ 1.0。
    说明: 使用 np.nanmin/max 完美兼容实拍数据中被挖空的 NaN 无效像素。
    """
    min_val = np.nanmin(matrix)
    max_val = np.nanmax(matrix)
    if max_val - min_val == 0: 
        return matrix - min_val 
    return (matrix - min_val) / (max_val - min_val)

def process_clean_img_file(image_path, roi_rect, target_resolution=(256, 256)):
    """
    【轨道 A：纯净数据流】处理前期的高位深纯净训练数据 (.img)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    x, y, w, h = roi_rect
    cropped_img = img[y:y+h, x:x+w]
    
    if len(cropped_img.shape) == 3: 
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
    float_matrix = cropped_img.astype(np.float32)
    resized_matrix = cv2.resize(float_matrix, target_resolution, interpolation=cv2.INTER_AREA)
    
    return normalize_physics_matrix(resized_matrix)

def process_dirty_png_file(image_path, roi_rect, ui_mask, target_resolution=(256, 256)):
    """
    【轨道 B：实拍数据流】处理后期的实拍实验截图 (.png)
    功能: 包含 NaN 物理屏蔽法，彻底剔除人类标记。
    """
    img = cv2.imread(str(image_path))
    x, y, w, h = roi_rect
    cropped_img = img[y:y+h, x:x+w]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    float_matrix = gray_img.astype(np.float32)
    resized_matrix = cv2.resize(float_matrix, target_resolution, interpolation=cv2.INTER_AREA)
    
    # 像素级物理屏蔽：被污染的像素不修图，直接置为 NaN
    resized_matrix[ui_mask] = np.nan 
    return normalize_physics_matrix(resized_matrix)

# =====================================================================
# 第四部分：数据对抗增强 (Adversarial Data Degradation)
# 对应策略：反向污染策略，用于锻炼后续的数学提取模型
# =====================================================================

def simulate_ui_degradation(clean_matrix):
    """
    功能: 【对纯净数据反向污染】。在完美的矩阵上，人为注入 NaN 破坏点。
    用法: 在 feature_extraction.py 测试高斯拟合算法抗干扰能力时调用。
    """
    degraded_matrix = clean_matrix.copy()
    h, w = degraded_matrix.shape
    
    # 在中心核心衍射区附近，随机画两条宽3像素的“十字准星”NaN污染带
    center_y, center_x = h // 2, w // 2
    degraded_matrix[center_y-1:center_y+2, :] = np.nan
    degraded_matrix[:, center_x-1:center_x+2] = np.nan
    
    return degraded_matrix