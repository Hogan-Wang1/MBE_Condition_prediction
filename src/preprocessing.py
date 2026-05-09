 # 图像预处理

import cv2
import numpy as np
from pathlib import Path
import os

# =====================================================================
# 第一部分：批次侦察与控制流辅助 (Strategies 1 & 2)
# =====================================================================

def analyze_batch_context(input_folder):
    """
    功能: 批次侦察兵。扫描输入文件夹，判断这是纯净训练批次还是实拍脏数据批次，并检查底片。
    返回: 字典，包含批次类型、首张图片路径、以及掩膜底片路径（如果有）。
    """
    folder_path = Path(input_folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
    # 获取所有图片文件
    all_files = list(folder_path.glob("*.*"))
    image_files = [f for f in all_files if f.suffix.lower() in ['.img', '.png', '.jpg']]
    
    if not image_files:
        raise ValueError(f"在 {folder_path} 中没有找到任何图片文件！")
        
    # 策略一：基于扩展名的硬路由
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
    
    elif ext == '.png':
        context["batch_type"] = "dirty"
        # 策略二：批次级上下文感知 (寻找掩膜底片)
        template_path = folder_path / "ui_mask_template.png"
        if not template_path.exists():
            raise FileNotFoundError(
                f"\n🚨 致命错误：这是 .png 实拍批次，但在文件夹中找不到名为 'ui_mask_template.png' 的全黑底片！\n"
                f"请放入底片后再运行程序。"
            )
        context["template_path"] = template_path
        print(f"🔍 侦察完毕：检测到 {context['total_count']} 张 .png 实拍数据，且 UI 底片就绪。")
        
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
        
    return context

# =====================================================================
# 第二部分：标定与掩膜生成 (Calibration & Masking)
# =====================================================================

def get_roi_from_first_frame(image_path):
    """弹出一个 GUI 窗口，让用户手工框选核心的衍射条纹区域。"""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None: raise ValueError(f"无法读取图片: {image_path}")

    # 如果是16-bit图像，为了能在屏幕上正常显示供人眼框选，需要临时归一化到8-bit
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    print("\n" + "="*40 + "\n📢 标定模式启动\n请框选核心衍射条纹并按【回车键】确认。\n" + "="*40 + "\n")
    roi_rect = cv2.selectROI("Select RHEED ROI", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    if roi_rect[2] == 0 or roi_rect[3] == 0: raise ValueError("标定取消或未画框！程序中断。")
    return roi_rect

def generate_ui_mask(template_path, roi_rect, target_resolution=(256, 256)):
    """读取全黑底片，提取像素级的绝对掩膜。"""
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    x, y, w, h = roi_rect
    cropped_template = template[y:y+h, x:x+w]
    # 必须用最近邻插值保持边界锐利
    resized_template = cv2.resize(cropped_template, target_resolution, interpolation=cv2.INTER_NEAREST)
    ui_mask = resized_template > 10
    return ui_mask

# =====================================================================
# 第三部分：核心数学归一化与双轨加载器
# =====================================================================

def normalize_physics_matrix(matrix):
    """将任意数值范围的矩阵，强行映射到 0.0 ~ 1.0。自动跳过 NaN。"""
    min_val = np.nanmin(matrix)
    max_val = np.nanmax(matrix)
    if max_val - min_val == 0: return matrix - min_val 
    return (matrix - min_val) / (max_val - min_val)

def process_clean_img_file(image_path, roi_rect, target_resolution=(256, 256)):
    """【轨道 A】处理前期的纯净训练数据 (.img)"""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    x, y, w, h = roi_rect
    cropped_img = img[y:y+h, x:x+w]
    if len(cropped_img.shape) == 3: cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
    float_matrix = cropped_img.astype(np.float32)
    resized_matrix = cv2.resize(float_matrix, target_resolution, interpolation=cv2.INTER_AREA)
    return normalize_physics_matrix(resized_matrix)

def process_dirty_png_file(image_path, roi_rect, ui_mask, target_resolution=(256, 256)):
    """【轨道 B】处理后期的实拍实验数据 (.png)，注入 NaN 掩膜"""
    img = cv2.imread(str(image_path))
    x, y, w, h = roi_rect
    cropped_img = img[y:y+h, x:x+w]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    float_matrix = gray_img.astype(np.float32)
    resized_matrix = cv2.resize(float_matrix, target_resolution, interpolation=cv2.INTER_AREA)
    
    resized_matrix[ui_mask] = np.nan # 像素级物理屏蔽
    return normalize_physics_matrix(resized_matrix)