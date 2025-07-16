import nibabel as nib
import numpy as np


def modify_mask_label(input_file ,save_path):
    img = nib.load(input_file)
    # 获取图像数据
    data = img.get_fdata()

    data[(data >= 81)&(data<=80)] = 80
    data[data == 100] = 80

    data[data == 9] = 28
    data[(data >= 25)&(data <= 37)] = 28
    data[(data>=92)&(data<=117)] =28
    data[(data != 28) & (data != 80)] = 3
    data = data.astype(np.int16)
    output_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(output_img, save_path)

if __name__ == '__main__':
    # 示例使用
    input_file = r"E:\lyh\sample_data_5_people_2\0\person_ct\杨邦杰_2223439_185206\YangBangJie.nii.gz"
    save_path = r"E:\lyh\sample_data_5_people_2\0\person_ct\杨邦杰_2223439_185206\YangBangJie_deal.nii.gz"
    modify_mask_label(input_file, save_path)