import os
from data.base_dataset import BaseDataset, get_transform,get_params
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # 初始化 transform_A 和 transform_B（记得导入 get_transform）
        self.transform_A = get_transform(opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(opt, grayscale=(self.output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)
        # B_path = self.B_paths[index_B]
        
        #玩全配对
        # A_path = self.A_paths[index % self.A_size]
        # A_filename = os.path.basename(A_path)
        # 查找 B 中同名文件
        # B_path = os.path.join(self.dir_B, A_filename)
        
        # # 安全性检查
        # if not os.path.exists(B_path):
        #     raise FileNotFoundError(f'Cannot find matching B image for {A_filename}')

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        # # apply image transformation
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)

        # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        
        #乱序但额外加载配对图
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[random.randint(0, self.B_size - 1)]
        
        # 加载对齐图像（文件名相同）
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[random.randint(0, self.B_size - 1)]
        
        A_filename = os.path.basename(A_path)
        B_filename = os.path.basename(B_path)
        
        # 查找同名配对图
        paired_B_path = os.path.join(self.dir_B, A_filename)
        paired_A_path = os.path.join(self.dir_A, B_filename)
        
        # 加载原始图
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # 获取共享 transform 参数
        params = get_params(self.opt, A_img.size)

        # 构建 transform（使用共享 params）
        transform_A = get_transform(self.opt, params, grayscale=(self.input_nc == 1))
        transform_B = get_transform(self.opt, params, grayscale=(self.output_nc == 1))

        # 应用 transform
        input_A = transform_A(A_img)
        input_B = transform_B(B_img)

        # 加载配对图（如果存在）
        if os.path.exists(paired_B_path):
            paired_B_img = Image.open(paired_B_path).convert('RGB')
            input_paired_B = transform_B(paired_B_img)
        else:
            input_paired_B = None
        
        if os.path.exists(paired_A_path):
            paired_A_img = Image.open(paired_A_path).convert('RGB')
            input_paired_A = transform_A(paired_A_img)
        else:
            input_paired_A = None
        
        # 构建输出 dict
        output = {
            'A': input_A,
            'B': input_B,
            'A_paths': A_path,
            'B_paths': B_path,
        }
        
        # 只在训练阶段返回 paired 图（防止 test 阶段出错）
        if self.opt.isTrain:
            # 或者：你可以加一个 self.opt.use_paired 之类的 flag 控制更细粒度
            if input_paired_B is not None:
                output['paired_B'] = input_paired_B
            if input_paired_A is not None:
                output['paired_A'] = input_paired_A
        
        return output


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
