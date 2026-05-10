"""
可视化模块
职责：
  - 基于时空立方体与有效性掩膜，生成报告图表：
      * Kymograph（空间-时间切片）
      * 平均衍射强度随时间变化曲线
      * STFT 频谱图（频率随时间变化）
  - 所有绘图函数均接受已处理好的数据，不直接接触原始文件。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_visualizations(spacetime_cube, validity_mask, output_dir, start_frame=1):
    """
    为一次实验生成完整的可视化报告。
    参数:
        spacetime_cube : np.ndarray, (T, H, W) 已归一化时空立方体
        validity_mask  : np.ndarray, 同上形状，True 为有效像素
        output_dir     : 输出文件夹路径
        start_frame    : 第一帧的真实序号（用于横轴标注）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    T, H, W = spacetime_cube.shape

    # ---- 1. Kymograph（中心水平线时空切片）----
    _save_kymograph(spacetime_cube, validity_mask, output_dir, start_frame)

    # ---- 2. 平均强度曲线 ----
    mean_intensity = _compute_mean_intensity(spacetime_cube, validity_mask)
    _save_intensity_curve(mean_intensity, output_dir, start_frame)

    # ---- 3. STFT 频谱图 ----
    if T > 128:
        _save_stft_spectrogram(mean_intensity, output_dir, start_frame)
    else:
        print("ℹ️ 帧数较少，跳过 STFT 频谱图生成。")

    print(f"📈 可视化报告已保存至 {output_dir}")

# -------------------------------------------------------------------
# 内部绘图辅助函数
# -------------------------------------------------------------------

def _save_kymograph(cube, mask, output_dir, start_frame):
    """绘制并保存中心水平线的 kymograph"""
    T, H, W = cube.shape
    center_line = H // 2
    kymo_data = cube[:, center_line, :]  # (T, W)
    kymo_valid = mask[:, center_line, :]
    # 将无效像素置 0 以便显示（原始 NaN 会显示为空白）
    kymo_plot = np.where(kymo_valid, kymo_data, 0)

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(kymo_plot.T, aspect='auto', origin='lower', cmap='inferno',
                   extent=[start_frame, start_frame + T - 1, 0, W])
    plt.colorbar(im, ax=ax, label='归一化强度')
    ax.set_xlabel('帧序号')
    ax.set_ylabel('空间位置 (像素)')
    ax.set_title('RHEED Kymograph (中心水平线)')
    fig.tight_layout()
    fig.savefig(output_dir / 'kymograph.png', dpi=150)
    plt.close(fig)

def _compute_mean_intensity(cube, mask):
    """计算每帧排除 NaN 后的空间平均强度"""
    # 避免除以零
    valid_counts = np.sum(mask, axis=(1, 2))
    valid_counts = np.where(valid_counts == 0, 1, valid_counts)  # 若全帧无效则填充
    mean_intensity = np.sum(cube * mask, axis=(1, 2)) / valid_counts
    return mean_intensity

def _save_intensity_curve(mean_intensity, output_dir, start_frame):
    """绘制平均强度随时间变化曲线"""
    T = len(mean_intensity)
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(range(start_frame, start_frame + T), mean_intensity, linewidth=0.8)
    ax.set_xlabel('帧序号')
    ax.set_ylabel('归一化平均强度')
    ax.set_title('平均衍射强度随时间变化')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'mean_intensity.png', dpi=150)
    plt.close(fig)

def _save_stft_spectrogram(mean_intensity, output_dir, start_frame):
    """绘制短时傅里叶变换频谱图"""
    from scipy.signal import spectrogram

    # 去除趋势，只保留振荡分量
    signal = mean_intensity - np.nanmean(mean_intensity)
    # 采样频率设为 1 Hz（帧/秒）
    f, t_seg, Sxx = spectrogram(signal, fs=1,
                                nperseg=min(256, len(signal) // 2),
                                noverlap=min(200, len(signal) // 4))
    fig, ax = plt.subplots(figsize=(15, 5))
    # 转成分贝并避免 log(0)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im = ax.pcolormesh(t_seg + start_frame, f, Sxx_db,
                       shading='gouraud', cmap='viridis')
    plt.colorbar(im, ax=ax, label='功率谱密度 (dB)')
    ax.set_ylabel('频率 (Hz)')
    ax.set_xlabel('帧序号')
    ax.set_title('强度振荡 STFT 频谱图')
    fig.tight_layout()
    fig.savefig(output_dir / 'stft_spectrogram.png', dpi=150)
    plt.close(fig)