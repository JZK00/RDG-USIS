import numpy as np
import nibabel as nib
import pickle
import json
from scipy.signal import fftconvolve
import cv2
import count_line as cl
import os
import matplotlib.pyplot as plt
import shutil
import deal_ct_mask_pipe as dcp

is_show = True
f = 1
i0 = 200
pixel_spacing = 0.6640625
result_image_shape = (512,512)
result_image_center = (250, 0)
psf_sigma = 3  # 高斯核的标准差
focus_center = (250, 250)
sigma_x = 2.5
sigma_y = 2.5
z_pixel_spacing = 0.79999999
lines_count = 128
origin = (300, 600)
depth = 310
probe_width = 66/180*np.pi
k = 2
psf_size = 4
sample = 1
needle_length = 200 # 单位mm

path_head_common = r"./"
scatterers_point_path = os.path.join(path_head_common, "scatterers.npy")
scatterers1 = np.load(scatterers_point_path)
tissue_param_path = os.path.join(path_head_common, "tissue_sound_parameter.json")


def find_rotated_vector(v1, normal, angle):
    e1 = v1 / np.linalg.norm(v1)
    # 计算 e2 = normal × e1
    e2 = np.cross(normal, e1)
    if np.linalg.norm(e2) == 0:
        raise ValueError("v1 与法向量平行，无法计算正交基")
    e2 = e2 / np.linalg.norm(e2)
    # 旋转角度 a
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # 计算旋转后的向量
    v2 = cos_a * e1 + sin_a * e2
    return v2

def generate_gaussian_psf(size, sigma):
    """
    创建一维高斯 PSF。
    :param size: 高斯核的大小（窗口宽度）
    :param sigma: 高斯函数的标准差
    :return: 一维高斯核
    """
    x = np.linspace(-size // 2, size // 2, size)
    psf = np.exp(-x ** 2 / (2 * sigma ** 2))
    psf /= np.sum(psf)  # 归一化
    return psf

def depth_dependent_convolution(image, sigma_min, sigma_max, psf_size=15):
    """
    根据深度对图像的每一行使用不同的高斯 PSF 进行卷积。
    :param image: 输入图像（二维数组）
    :param sigma_min: 浅层（顶部行）的最小高斯标准差
    :param sigma_max: 深层（底部行）的最大高斯标准差
    :param psf_size: 高斯核大小
    :return: 卷积后的图像
    """
    rows, cols = image.shape
    output_image = np.zeros_like(image)
    for i in range(rows):
        # 动态计算当前行的 sigma，线性插值
        sigma = sigma_min + (sigma_max - sigma_min) * (i / rows)
        # 生成对应的高斯核
        psf = generate_gaussian_psf(psf_size, sigma)
        # 对当前行进行卷积
        output_image[i, :] = np.convolve(image[i, :], psf, mode='same')
    return output_image

def post_process(image,psf_size):
    result_image = np.where(np.isnan(image), 0, image)
    result_image = np.log(1 + np.abs(result_image * 600))
    x = np.linspace(-psf_size // 2, psf_size // 2, psf_size)
    y = np.linspace(-psf_size // 2, psf_size // 2, psf_size)
    x, y = np.meshgrid(x, y)
    psf = np.exp(-0.5 * ((x ** 2) / sigma_x ** 2 + (y ** 2) / sigma_y ** 2))
    psf /= np.sum(psf)
    # 模拟超声图像
    img_data = result_image * scatterers1
    ultrasound_image = fftconvolve(img_data, psf, mode='same')
    bright_adjust = 3
    normalized_image = 255 * ultrasound_image / bright_adjust
    result_image = normalized_image.astype(np.uint8)  # 转换为 uint8 类型
    result_image = resize_image(result_image)  # 提高分辨率
    sigma_min = 1  # 浅层高斯核宽度
    sigma_max = 16  # 深层高斯核宽度
    psf_size = 31  # 高斯核大小
    # 对图像进行深度依赖的卷积处理
    processed_image = depth_dependent_convolution(result_image, sigma_min, sigma_max, psf_size)
    return processed_image

def create_polar_coordinates_lines(dir_vectory, nor_vectory, image, lines_count, origin_mask_point, depth, probe_width, k, psf_size, sample,tissue_param_data,
                                                              scatterers_energy_arr,
                                                              closest_point):
    # 角度范围
    numbers = np.linspace(-probe_width / 2, probe_width / 2, lines_count)
    result_image = np.zeros(result_image_shape)
    origin_length = 0
    for theta in numbers:
        rotated_vector = find_rotated_vector(dir_vectory, nor_vectory, theta)
        result_image = cl.create_one_line(result_image, theta, rotated_vector, image, origin_mask_point, depth, origin_length, k, [], [], [], sample, tissue_param_data, scatterers_energy_arr, closest_point)
    processed_image = post_process(result_image, psf_size)
    return processed_image

def resize_image(image):
    height, width = image.shape[:2]
    new_width = int(width * 3.2)
    new_height = int(height * 3.2)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


def main():
    image_store = f"./data/"
    path_head = f"./load_data"
    if not os.path.exists(path_head):
        os.makedirs(path_head)
    if not os.path.exists(image_store):
        os.makedirs(image_store)
    scatterers_energy_path = os.path.join(path_head, "scatterers_energy.npy")
    nii_name = "ct_mask.nii.gz"
    nii_file = os.path.join(path_head, nii_name)
    # 生成
    dcp.main(nii_file, path_head)
    # 加载数据
    close_points = os.path.join(path_head, "closest_point_3d.pkl")
    with open(close_points, 'rb') as file:  # 使用 'rb' 模式读取二进制文件
        closest_point = pickle.load(file)
    scatterers_energy_arr = np.load(scatterers_energy_path)
    img = nib.load(nii_file)
    data = img.get_fdata()
    with open(tissue_param_path, 'r', encoding='utf-8') as file:
        tissue_param_data = json.load(file)

    for i in range(0, data.shape[2]):
        point_m = np.array([0, 255 * pixel_spacing, i * z_pixel_spacing]).astype(np.float64)
        normal_vectory = np.array([0, 0, -1]).astype(np.float64)
        direction_vectory = np.array([1, 0, 0]).astype(np.float64)
        result_image = create_polar_coordinates_lines(direction_vectory, normal_vectory,
                                                      data, lines_count, point_m, depth,
                                                      probe_width, k,
                                                      psf_size, sample, tissue_param_data,
                                                      scatterers_energy_arr,
                                                      closest_point)
        print("point:", point_m)
        result_image = cv2.resize(result_image, (512, 512), interpolation=cv2.INTER_LINEAR)
        filename = f"image_{i:05d}.png"
        store_path = os.path.join(image_store, filename)
        zeros = np.zeros(result_image.shape)
        zeros[:512 - 154, :] = result_image[154:, :]
        cv2.imwrite(store_path, zeros)
    shutil.rmtree(path_head)

if __name__ == "__main__":
    main()