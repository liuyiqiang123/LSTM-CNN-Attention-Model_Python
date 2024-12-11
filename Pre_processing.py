from scipy.signal import savgol_filter
import numpy as np


def rf_transform(data):
    return np.log10(1 / data.astype(float))


def SG_derivative(data, window_length=101, polyorder=3):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, deriv=1, axis=1)


def SNV(spectra):
    """
    对光谱数据进行标准正态变量转换(SNV)

    参数:
        spectra: ndarray, shape=(n_samples, n_wavelengths)
                输入光谱矩阵
    返回:
        snv_spectra: ndarray, shape=(n_samples, n_wavelengths)
                   SNV预处理后的光谱矩阵
    """
    # 计算每个样本光谱的均值
    mean = np.mean(spectra, axis=1, keepdims=True)
    # 计算每个样本光谱的标准差
    std = np.std(spectra, axis=1, keepdims=True, ddof=1)
    # SNV转换
    snv_spectra = (spectra - mean) / std

    return snv_spectra

def msc(data):
    mean_spectrum = np.mean(data, axis=0)
    corrected_data = np.empty_like(data)
    for i in range(data.shape[0]):
        fit = np.polyfit(mean_spectrum, data[i, :], 1, full=True)
        corrected_data[i, :] = (data[i, :] - fit[0][1]) / fit[0][0]
    return corrected_data


def lrd(data, window_length=11, polyorder=2):
    # 对数据进行倒数变换
    reciprocal_data = 1.0 / data
    # 取对数
    log_data = np.log(reciprocal_data)
    # 使用 Savitzky-Golay 滤波器计算一阶导数
    derivative_data = savgol_filter(log_data, window_length=window_length, polyorder=polyorder, deriv=1, axis=1)

    return derivative_data


def continuum_removal(data):
    corrected_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        hull = np.maximum.accumulate(data[i, :])
        corrected_data[i, :] = data[i, :] / hull
    return corrected_data
