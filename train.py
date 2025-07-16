import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    ngpus_per_node = torch.cuda.device_count() #判断是否有多卡
    dist.init_process_group(backend="nccl")
    print("使用多卡数量ngpus_per_node=",ngpus_per_node)
    rank = dist.get_rank()
    seed = rank
    torch.manual_seed(seed)
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    opt.device_id = device
    model = create_model(opt)  # 创建模型
    model.setup(opt)
    if opt.use_distributed and ngpus_per_node > 1:  # 使用DDP模式
        print("Using {} GPUs".format(ngpus_per_node))
        print("=============+++++++++++++ 当前的device",device)
        model.to(device)
        model = DDP(model, device_ids=[device], find_unused_parameters=True)  # 包装模型
    else:
        model = create_model(opt)  # 创建模型
        model.setup(opt)
    # 2.修改dataset
    dataset,train_sampler = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        if opt.use_distributed:  # 确保数据同步
            dist.barrier()
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()
        # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        if isinstance(model, DDP):
            model.module.update_learning_rate()  # 访问原始模型并调用 update_learning_rate 方法
        else:
            model.update_learning_rate()  # 如果不是 DDP，直接调用模型的 update_learning_rate 方法
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if opt.use_distributed: # 每个GPU读取不同的数据
                train_sampler.set_epoch(epoch)
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if isinstance(model, DDP):
                model.module.set_input(data)  # unpack data from dataset and apply preprocessing
                model.module.optimize_parameters()
            else:
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()
                # calculate loss functions, get gradients, update network weights
            if opt.use_distributed:  # 确保数据同步
                dist.barrier()
            if total_iters % opt.display_freq == 0 and device == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                if isinstance(model, DDP):
                    model.module.compute_visuals()
                    visualizer.display_current_results(model.module.get_current_visuals(), epoch, save_result)
                else:
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                if isinstance(model, DDP):
                    losses = model.module.get_current_losses()
                else:
                    losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data,device)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0 and device == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                if isinstance(model, DDP):
                    model.module.save_networks(save_suffix)
                else:
                    model.save_networks(save_suffix)
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0 and device == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            if isinstance(model, DDP):
                model.module.save_networks('latest')
                model.module.save_networks(epoch)
            else:
                model.save_networks('latest')
                model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
