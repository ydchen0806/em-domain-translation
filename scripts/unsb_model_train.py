import time
import torch
import os
import random
from PIL import Image
import imageio
import numpy as np
from pathlib import Path
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from options.train_options import TrainOptions

from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from data import create_dataset

from monai.data import DataLoader
from data.dataset_lb import CondTargetDataset_lb
from networks_copy.util import create_model


# 沿用UNSB的框架
import warnings
warnings.filterwarnings("ignore")

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def losses_saver(tensorboard_writer, losses, epoch):
    for k, v in losses.items(): # losses['G']= xxx . k=
        tensorboard_writer.add_scalar('loss/loss_%s'%k, v, epoch)
        
def visual_saver(data, save_path,epoch):
    img_path=os.path.join(save_path, 'img')
    video_path=os.path.join(save_path, 'video')
    Path(img_path).mkdir(parents=True, exist_ok=True)
    Path(video_path).mkdir(parents=True, exist_ok=True)
    '''
    ('real_A', real_A), ('fake_B', fake_B),
                            ('real_B', real_B),
                            ('idt_B', idt_B)])#
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                            ('real_B', real_B)])
    
    '''
    img = data['real_A']  # torch.Size([1, 1, 160, 160, 18])
    img = img.squeeze().squeeze().cpu().detach().numpy()
    img = img * 255.0
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8) 
    img = np.transpose(img, (2, 1, 0))

    imgB = data['real_B']
    imgB = imgB.squeeze().squeeze().cpu().detach().numpy()
    imgB = imgB * 255.0
    imgB = np.clip(imgB, 0, 255)
    imgB = imgB.astype(np.uint8)
    imgB = np.transpose(imgB, (2, 1, 0))

    fB = data['fake_B']
    fB = fB.squeeze().squeeze().cpu().detach().numpy()
    fB = fB * 255.0
    fB = np.clip(fB, 0, 255)
    fB = fB.astype(np.uint8)
    fB = np.transpose(fB, (2, 1, 0))

    
    # [32, 160, 160]
    k = random.randint(0, img.shape[0]-1)
    Image.fromarray(img[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_rA.png'))
    Image.fromarray(imgB[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_rB.png'))
    Image.fromarray(fB[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_fB.png'))
 
    imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_rA.mp4'), img, fps=6)
    imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_rB.mp4'), imgB, fps=6)
    imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_fB.mp4'), fB, fps=6)

    if data['idt_B'] is not None:
        idt_B = data['idt_B']
        idt_B = idt_B.squeeze().squeeze().cpu().detach().numpy()
        idt_B = idt_B * 255.0
        idt_B = np.clip(idt_B, 0, 255)
        idt_B = idt_B.astype(np.uint8)
        idt_B = np.transpose(idt_B, (2, 1, 0))
        Image.fromarray(idt_B[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_idtB.png'))
        imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_idtB.mp4'), idt_B, fps=6)



if __name__ == '__main__':
    
    opt = TrainOptions().parse()   # get training options
    # CondTargetDataset_lb, 长度为opt.dataloader_len, strong_aug=False
    local_rank, world_size, device = initialize_distributed(len(opt.gpu_ids))
    logger = setup_logging("training")
    logger.info(f"Using {device} of {world_size}")



    if local_rank == 0:
        # device='cuda:0'
        save_path = os.path.join(opt.output_dir, opt.name)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(save_path)

        logger.info(f"Tensorboard event will be saved as {save_path}.")
        
    # dataset = create_dataset(opt)
    dataset = CondTargetDataset_lb(src_dir='./local_data/AC3_AC4/', src_dataset='AC4_inputs.h5',\
                                        tgt_dir='./local_data/CREMI', tgt_dataset='cremiC_inputs_interp.h5', \
                                        length=opt.dataloader_len,src_aug=False,strong_aug=False)
    # dataset = CondTargetDataset_lb(src_dir='./local_data/CREMI', src_dataset='cremiC_inputs_interp.h5',\
    #                                       tgt_dir='./local_data/AC3_AC4/', tgt_dataset='AC4_inputs.h5', \
    #                                         length=opt.dataloader_len,src_aug=False,strong_aug=False)
    dataloader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4, #int(opt.num_threads),
            drop_last=True, # if opt.isTrain else False,
        )
    # dataset2 = create_dataset(opt)
    dataset_size = len(dataloader)    # data / batchsize
    model = create_model(opt,device)      # sb_model.py

    # print('The number of training images = %d' % dataset_size)

    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    # opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations
    best_loss = 1e4
    epoch_loss_dict = {name: 0.0 for name in model.loss_names}

    # optimize_time = 0.1
    
    times = []
    prev_time = time.time()
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        # iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        # dataset.set_epoch(epoch)
        # dataset2和dataset是一样的调用，应该是里面有data=[A,B]
        # for i, (data,data2) in enumerate(zip(dataset,dataset2)):  # inner loop within one epoch
        for step, data in enumerate(dataloader): # 现在改成batchsize=2,之后再分成一个A和一个B
            iter_start_time = time.time()  # timer for computation per iteration
            
            # ？？这一步的作用
            # if total_iters % opt.print_freq == 0:
            #     t_data = iter_start_time - iter_data_time

            batch_size = opt.batch_size #data["A"].size(0)
            total_iters += batch_size # 麻烦，，16【改成根据epoch！！！】
            epoch_iter += batch_size
            
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            optimize_start_time = time.time()
            # 训练的开头，会先加载一下netF，用于赋值。
            # 新加的： 计算scale_factor，加载netG
            if epoch == opt.epoch_count and step == 0: 
                model.data_dependent_initialize(data) # set_input+forward【会设置realA只有1】
                model.setup(opt,device)      #  会到这一步
                # model.parallelize()
            
            torch.cuda.empty_cache()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch=epoch, thres = -1, update_freq=1)   # calculate loss functions, get gradients, update network weights
            # epoch=epoch, thres = 20, update_freq=2
            
            if local_rank == 0:
                batches_done = step + 1
                batches_left = dataset_size - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                losses = model.get_current_losses() # 默认考虑了batchsize
                for name in model.loss_names:
                    epoch_loss_dict[name] += losses[name]
                print(
                        "\r[Epoch %d/%d] [Batch %d/%d] [loss: %.4f] ETA: %s "
                        % (
                            epoch,
                            opt.n_epochs + opt.n_epochs_decay,
                            step + 1,
                            dataset_size,
                            losses['G'], # 单纯一个step的loss
                            time_left,
                        )
                    )
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            # optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time
        
        torch.distributed.barrier()
        
        if epoch>=350:
            opt.display_freq = 1
        elif epoch>=100:
            opt.display_freq = 2
        
        if local_rank == 0: # epoch_loss_dict
            for name in model.loss_names:
                epoch_loss_dict[name] /= dataset_size
            losses_saver(tensorboard_writer, epoch_loss_dict, epoch)
            if epoch_loss_dict['G'] < best_loss:
                best_loss = epoch_loss_dict['G']
                model.save_networks(epoch, best_loss)
            if epoch % opt.display_freq == 0: # 5
                torch.cuda.empty_cache()
                
                imgdict = model.get_current_visuals()
                visual_saver(imgdict, save_path, epoch)

                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch, epoch_loss_dict['G'])

            print('End of epoch %d / %d \t Epoch LossG = %.4f \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, epoch_loss_dict['G'], time.time() - epoch_start_time))
            epoch_loss_dict = {name: 0.0 for name in model.loss_names}

        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        
        torch.cuda.empty_cache()
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()