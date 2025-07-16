# RDG-USIS
Real-Time, Dynamic, and Highly Generalizable Ultrasound Image Simulation-Guided Procedure  Training System for Musculoskeletal Minimally Invasive Treatment. 

## Introduction
Citation: XXX

Here, we propose a Real-time, Dynamic, and highly Generalizable UltraSound Image Simulation (RDG-USIS) algorithm, specifically designed to enhance training in minimally invasive procedures.

The RDG-USIS:
![本地图片描述](Figures/Fig1.bmp)
<img width="300" alt="452a6e..." src="https://github.com/user-attachments/assets/691ee41b-b300-4b8a-99a7-8f6c873aa186"/>
<img width="300" alt="a0ebd3..." src="https://github.com/user-attachments/assets/ddc890e6-f153-4f04-a64d-740460249d3e"/>

Our developed ultrasound image simulation-guided minimally invasive procedure training system integrates the proposed RDG-USIS algorithm. It generates high-quality ultrasound images from CT scans (see module indicated by the red circle). It supports real-time, dynamic alignment with other multimodal imaging data, significantly enhancing 3D spatial understanding and surgical accuracy during ultrasound-guided training. 

## How to Start Convolutional Simulation Of Ultrasound
python cov_img/get_sim_us.py

## How to Start Project
Install dependencies:

pip install -r requirements.txt

The project is only compatible with multi-GPU DDP mode for training.

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=12345 --nnodes=1 --nproc_per_node=4 train.py  --dataroot ./datasets/test --name test --model cycle_gan --use_distributed  --lambda_ssim 5

## Dataset
After the article is accepted, we will open-source the high-quality US-CT dataset that we have designed and collected, which will have a positive impact on the community.

