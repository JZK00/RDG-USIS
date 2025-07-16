# 将ct分割的mask文件进行流水线处理
import nibabel as nib
import numpy as np
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from skimage import measure

def set_scatters(nii_file,save_path):
    img = nib.load(nii_file)
    data = img.get_fdata()
    data1 = data.astype(int)
    data1[data1==24] = 36
    data[data1 == 3] = np.random.uniform(0.00015, 0.0003, np.sum(data1 == 3))
    data[data1 == 81] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 81))
    data[data1 == 82] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 82))
    data[data1 == 83] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 83))
    data[data1 == 84] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 84))
    data[data1 == 85] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 85))
    data[data1 == 86] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 86))
    data[data1 == 87] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 87))
    data[data1 == 88] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 88))
    data[data1 == 89] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 89))
    data[data1 == 80] = np.random.uniform(0.00051, 0.00053, np.sum(data1 == 80))
    data[data1 == 9] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 9))
    data[data1 == 25] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 25))
    data[data1 == 26] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 26))
    data[data1 == 27] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 27))
    data[data1 == 28] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 28))
    data[data1 == 29] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 29))
    data[data1 == 30] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 30))
    data[data1 == 31] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 31))
    data[data1 == 32] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 32))
    data[data1 == 33] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 33))
    data[data1 == 34] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 34))
    data[data1 == 35] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 35))
    data[data1 == 36] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 36))
    data[data1 == 37] = np.random.uniform(0.00035, 0.00037, np.sum(data1 == 37))
    # 保存修改后的数据
    np.save(save_path, data)

# 导出相邻点集合
def cache_closest_n_points_3d(nii_file, save_path):
    n = 48
    # 假设您的3D数据存储在NIfTI文件中
    img = nib.load(nii_file)
    mask = img.get_fdata()
    # 将mask转换为二值化数据，假设要处理的目标标签不是12
    mask = np.uint8(mask != 3)
    # 获取3D边界点（表面体素）
    # 使用Marching Cubes算法提取等值面
    verts, faces, normals, values = measure.marching_cubes(mask, level=0)
    boundary_points = verts  # 边界点的坐标
    # 创建KD树以加速最近邻搜索
    nbrs = NearestNeighbors(n_neighbors=n + 1, algorithm='auto').fit(boundary_points)
    distances, indices = nbrs.kneighbors(boundary_points)
    # 创建映射字典，将每个边界点的索引映射到其最近的n个边界点的索引
    out_data = {}
    for idx, (point_indices, point_distances) in enumerate(zip(indices, distances)):
        # 跳过第一个索引，因为它是点自身
        closest_indices = point_indices[1:]  # 最近的n个点的索引
        closest_points = boundary_points[closest_indices]
        # 将结果保存到字典中，键是当前点的坐标，值是最近n个点的坐标
        current_point = tuple(boundary_points[idx])
        print(current_point)
        out_data[current_point] = closest_points
    # 保存结果到文件
    save_name = "closest_point_3d.pkl"
    cache_file = os.path.join(save_path, save_name)
    with open(cache_file, 'wb') as file:
        pickle.dump(out_data, file)
    print("最近的点已成功缓存。")

def main(nii_file,store_path):
    scatters_name = "scatterers_energy.npy"
    scatters_path = os.path.join(store_path, scatters_name)
    set_scatters(nii_file, scatters_path)
    cache_closest_n_points_3d(nii_file, store_path)

if __name__ == '__main__':
    main()

