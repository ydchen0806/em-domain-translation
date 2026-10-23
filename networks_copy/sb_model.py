import numpy as np
import torch
import argparse
import json
import monai
import torch.distributed as dist
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from .autoencoderkl_maisi import load_autoencoder

from .patchnce import PatchNCELoss
import utils.utils as util

# torch.autograd.set_detect_anomaly(True)

def apply_autoencoder(
        autoencoder,
        inputs, # [b, 1, 160, 160, z]  -> [b, 4, 40, 40, z]
        use_vqgan=False,
        recon=False,
        scale_factor=None):
    # decode
    if recon: 
        if use_vqgan:
            scale_factor = 1.0
            # z = ((inputs.permute(0,1,4,3,2)/scale_factor) * (autoencoder.codebook.embeddings.max() -
            #                                         autoencoder.codebook.embeddings.min())) + autoencoder.codebook.embeddings.min()
            z = (((inputs.permute(0,1,4,3,2) + 1.0) / 2.0) * (autoencoder.codebook.embeddings.max() -
                                                    autoencoder.codebook.embeddings.min())) + autoencoder.codebook.embeddings.min()
            # z = inputs.permute(0,1,4,3,2) # /scale_factor
            return autoencoder.decode(z,quantize=True).permute(0,1,4,3,2)
        else:
            return autoencoder(inputs) # Recon_model
    # encode
    if use_vqgan:
        z = autoencoder.encode(inputs.permute(0,1,4,3,2),
                                               quantize=False)
        # z = ((z - autoencoder.codebook.embeddings.min()) /
        #              (autoencoder.codebook.embeddings.max() -
        #               autoencoder.codebook.embeddings.min())).detach()

        z = ((z - autoencoder.codebook.embeddings.min()) /
                     (autoencoder.codebook.embeddings.max() -
                      autoencoder.codebook.embeddings.min())) * 2.0 - 1.0
        return z.permute(0,1,4,3,2)
    else: # 【注意用vae的话要同步修改dataset_lb和single_dataset为[32,160,160]，[16]不行】
        return autoencoder.encode_stage_2_inputs(inputs)


def calculate_scale_factor(
        images,
        cond,
        autoencoder,
        device, 
        use_vqgan=False) -> torch.Tensor:

    # 如果每张gpu都计算(b=8, xxx)会太大
    if images.shape[0]>1:
        images = images[:1]
        cond = cond[:1]
    z = images.to(device) # [b=1, 1, 160, 160, 32]
    zc = cond.to(device)
    
    with torch.amp.autocast("cuda"):
        # batch = torch.from_numpy(images).float().to(device).unsqueeze(0).unsqueeze(0)
        z = apply_autoencoder(autoencoder, z, use_vqgan=use_vqgan) # [b, 4, 40, 40, 8]
        z=z.cpu().detach().numpy()
        z = torch.from_numpy(z).to(device)

        zc = apply_autoencoder(autoencoder, zc, use_vqgan=use_vqgan)
        zc=zc.cpu().detach().numpy()
        zc = torch.from_numpy(zc).to(device)
    scale_factor = 1 / torch.std(z) # scale_factor: tensor(0.9126)
    scale_factor_cond = 1 / torch.std(zc)
    del z,zc
    print(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
        dist.all_reduce(scale_factor_cond, op=torch.distributed.ReduceOp.AVG)
    torch.cuda.empty_cache()
    return scale_factor,scale_factor_cond


class GCNModel(torch.nn.Module):
    def __init__(self, args, init_type, init_gain, device, checkpoint):
        super(GCNModel, self).__init__()

        # opt.init_type, opt.init_gain, self.device
        from scripts.utils import define_instance

        net = define_instance(args, "netG_def") #.to(device)
        self.net1=networks.init_net(net, init_type, init_gain, device, initialize_weights=True)


        self.net2 = define_instance(args, "controlnet_def").to(device)
        controlnet_ckpt = torch.load(checkpoint, map_location=device)
        monai.networks.utils.copy_model_state(self.net2, controlnet_ckpt["controlnet_state_dict"])

    def forward(self, input, t, z=None, cond=None, layers=[0,0,3,6,9,11,12,13], encode_only=False):
        '''
        feat = self.netG(
                x=input.clone().detach().requires_grad_(True),
                timesteps=torch.zeros_like(self.time_idx),
                context=z,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                layers=self.nce_layers, encode_only=True
            )
        '''
        down_block_res_samples, mid_block_res_sample = self.net2(
            x=input, timesteps=t, controlnet_cond=cond
        )
        output = self.net1(
            x=input,
            timesteps=t,
            context=z,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            layers=layers, encode_only=encode_only
        )
        return output


# 感觉是CycleGAN->CUT->UNSB
class SBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for SB model
        """
        parser.add_argument('--mode', type=str, default="sb", choices='(FastCUT, fastcut, sb)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='下面会修改成0.5weight for NCE loss: NCE(G(X), X)')
        #  
        parser.add_argument('--lambda_idt', type=float, default=10.0, help='identity loss')
        parser.add_argument('--lambda_SB', type=float, default=1.0, help='weight for SB loss')
        
        parser.add_argument('--idt', type=util.str2bool, nargs='?', const=True, default=False, help='use l1 loss for identity mapping')
        # 下面设置这个nce_idt为default=True
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,0,3,6,9,11,12,13', help='original:0,4,8,12,16,compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        # 
        # 【记得修改，vae】
        parser.add_argument('--weighted_loss', type=int, default=100, help='weight for aff 0')
        parser.add_argument('--use_vqgan', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--use_144', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--use_cn', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--vq_checkpoint',type=str, 
                            default='./local_data/exp_vqgan/MIX/CremiC+AC4_train_18_data80/lightning_logs/version_0/checkpoints/epoch=1359-step=33999-train/recon_loss=0.05.ckpt', help='path to checkpoint')
        parser.add_argument('--netG_checkpoint',type=str, 
                            default='./models/maisi/model_maisi_input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1_alternative.pt', help='path to checkpoint')
        parser.add_argument('--controlnet_checkpoint',type=str, 
                            default="./models/maisi/model_maisi_controlnet-20datasets-e20wl100fold0bc_noi_dia_fsize_current_alternative.pt", help='path to checkpoint')
        parser.add_argument('--lr_cn', type=float, default=1e-7, help='initial learning rate for adam')
        parser.add_argument('--vae_checkpoint',type=str, 
                            default='./models/vae_144_final/autoencoder_epoch198.pt', help='path to checkpoint')
        
        parser.add_argument('--json',type=str, 
                            default='./configs/unsb/config_unsb.json', help='path to checkpoint')
        parser.set_defaults(pool_size=0)  # no image pooling

        # 【注意！！】use_cn = unsb+controlnet同时一起训练
        parser.set_defaults(use_vqgan=True, use_144=True, use_cn=True)

        opt, _ = parser.parse_known_args()

        # opt.use_vqgan=True
        # opt.use_144=True

        # Set default parameters for CUT and FastCUT
        # 1.用cn的话不能用nce_idt，会OOM
        if opt.mode.lower() == "sb":
            parser.set_defaults(idt=True, nce_idt=False, lambda_NCE=0.5)
            # parser.set_defaults(nce_idt=True, lambda_NCE=0.5) # idt=True
        elif opt.mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.mode)

        return parser

    

    def __init__(self, opt,device=None):
        BaseModel.__init__(self, opt,device)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE','SB']
        self.visual_names = ['real_A','real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']
        if opt.idt:
            self.loss_names += ['idt'] # 这里的loss_names必须与loss_xxx完全一致
            self.visual_names += ['idt_B']
            self.criterionIdt = torch.nn.L1Loss()

        if self.isTrain: # autoencoder不能加在这里，这里是会固定保存模型的
            self.model_names = ['G', 'F', 'D','E']
            # if opt.use_cn:
            #     self.model_names += ['CN']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        # netG 生成器；netF特征提取器；netD 判别器；netE 判别器
        ## 【看看要不要用MAISI，因为要加载vae autoencoder, 然后要记得scale_factor(src和tgt)】
            # 但是那个模型加载是用json的
        
        # 【用json文件加载模型！！尝试】
        args = argparse.Namespace()
        with open(opt.json, "r") as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)
        if opt.use_vqgan:
            self.autoencoder = load_autoencoder(opt.vq_checkpoint,args,self.device,use_vqgan=opt.use_vqgan)
        else:
            self.autoencoder = load_autoencoder(opt.vae_checkpoint,args,self.device,use_vqgan=opt.use_vqgan)
        
        for p in self.autoencoder.parameters(): 
            p.requires_grad = False
        self.autoencoder.eval()
        
        self.scale, self.scale_cond = 1.0, 1.0

        # self.scaler = torch.amp.GradScaler("cuda")
        # self.amp=True

        # torch.set_float32_matmul_precision("highest")
        
        # 【再修改G，需要有时输出feature..】
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        if opt.use_cn:
            self.netG = GCNModel(args, opt.init_type, opt.init_gain, self.device, opt.controlnet_checkpoint)
        else:
            self.netG = networks.define_G(args, opt.init_type, opt.init_gain, self.device)
        # 但是之前diffusion的是预测噪声。。。
        #self.load_netG0(opt)
        if self.opt.lambda_NCE > 0.0:
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.device, opt)

        
        # 【weights-wjw 251209】
        if opt.weighted_loss > 1.0:
            self.weighted_loss = opt.weighted_loss
            self.weighted_loss_label = [0] # label = 0
        else:
            self.weighted_loss = 1.0

        # 【260206】
        # if opt.use_cn:
        #     # self.controlnet = load_controlnet(opt.controlnet_checkpoint, args, self.device)
        #     import monai
        #     from scripts.utils import define_instance

        #     self.netCN = define_instance(args, "controlnet_def").to(device)
        #     controlnet_ckpt = torch.load(opt.controlnet_checkpoint, map_location=device)
        #     # monai.networks.utils.copy_model_state(controlnet, self.netG.state_dict())
        #     ### 直接严格读取maisi的checkpoint是不行的
        #     # self.controlnet.load_state_dict(controlnet_ckpt["controlnet_state_dict"], strict=True)
        #     monai.networks.utils.copy_model_state(self.netCN, controlnet_ckpt["controlnet_state_dict"])
        # else:
        #     self.netCN = None


        if self.isTrain:
            # basic_cond, NLayerDiscriminator_ncsn (PatchGAN) 会把两个input拼接起来
            # D: input_ch=3/1, opt.ndf=64,  opt.output_nc=3/1
            # self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
            #                               opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD = networks.define_D(args, opt.init_type, opt.init_gain, self.device)
            self.netE = networks.define_E(args, opt.init_type, opt.init_gain, self.device)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            # self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)
            # if opt.use_cn:
            #     self.optimizer_CN = torch.optim.AdamW(params=self.netCN.parameters(), lr=opt.lr_cn)
            #     self.optimizers.append(self.optimizer_CN)
    

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        特征网络netF是根据netG的编码器部分的中间提取特征的形状来定义的。
        因此，netF的权重在第一次前馈传递时初始化一些输入图像.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        # image: [b, 1, 160, 160, 18]
        # 【不太懂这里到底是几个GPU训练的，vae的ddp会不会影响。debug】
        '''
        "image": imageout,
        "tgt":imageout_tgt,
        "cond":cond, # 无增强
        "lb":lb_affs,
        "weight":weightmap}
        '''
        # bs_per_gpu = 2
        # bs_per_gpu = data["image"].size(0) // max(len(self.opt.gpu_ids), 1)
        # self.set_input(data,data2)
        self.set_input(data,if_scale=True) # Batchsize分一半A/B=8, A2/B2=8
        # self.real_A = self.real_A[:bs_per_gpu]
        # self.real_B = self.real_B[:bs_per_gpu]
        # # 【这是增加的代码】 A和A2会有b=2
        # self.real_A2 = self.real_A2[:bs_per_gpu]
        # self.real_B2 = self.real_B2[:bs_per_gpu]
        
        

        self.forward()                     # compute fake images: G(A)
        # with torch.amp.autocast("cuda"):
        loss_G = self.compute_G_loss()
        loss_D = self.compute_D_loss() 
        loss_E = self.compute_E_loss()
        if self.opt.isTrain:
            # if self.amp:
            #     self.scaler.scale(loss_G).backward()
            #     self.scaler.scale(loss_D).backward()
            #     self.scaler.scale(loss_E).backward()
            # else:
            loss_G.backward()
            loss_D.backward()
            loss_E.backward()  

            # 优化 特征提取器F的参数
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self,epoch=1, thres = 50, update_freq=2):
        # forward
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        # if self.opt.use_cn:
        #     self.netCN.train()

        # --- 动态频率控制 ---
        # 前 50 epoch，每 3 次迭代更新一次 D 和 E
        update_limit = update_freq if epoch < thres else 1
        
        if getattr(self, 'step_count', 0) % update_limit == 0:
            # 更新 D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_D.backward()
            self.optimizer_D.step()

            # 更新 E
            self.set_requires_grad(self.netE, True)
            self.optimizer_E.zero_grad()
            self.loss_E = self.compute_E_loss()
            self.loss_E.backward()
            self.optimizer_E.step()
        
        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        self.set_requires_grad(self.netG, True)

        self.optimizer_G.zero_grad()

        # if self.opt.use_cn:
        #     self.set_requires_grad(self.netCN, True)
        #     self.optimizer_CN.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        # if self.opt.use_cn:
        #     self.optimizer_CN.step()

        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()  

        # --- 4. 【核心修改点】在这里进行强制同步和数值平准 ---
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size() # 这是实际使用的GPU数量
            
            torch.distributed.all_reduce(self.loss_G, op=torch.distributed.ReduceOp.SUM)
            self.loss_G /= world_size
            
            if hasattr(self, 'loss_D'):
                torch.distributed.all_reduce(self.loss_D, op=torch.distributed.ReduceOp.SUM)
                self.loss_D /= world_size
                
            if hasattr(self, 'loss_E'):
                torch.distributed.all_reduce(self.loss_E, op=torch.distributed.ReduceOp.SUM)
                self.loss_E /= world_size
        
        self.step_count = getattr(self, 'step_count', 0) + 1

        torch.cuda.empty_cache()

    def optimize_parameters_original(self):
        # forward
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        
        self.loss_D.backward()
        self.optimizer_D.step()

        # E
        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        
        self.loss_E.backward()
        self.optimizer_E.step()
        
        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        # self.set_requires_grad(self.netG, True)

        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()     
        
    def set_input(self, input,if_2=True, if_scale=False):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """

        
        # [b, 1, 160, 160, 32]. b=2*8=16
        # 这里要做的是分一半
        divid_batch = input['image'].size(0) // 2 # 8
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['image' if AtoB else 'cond'][:divid_batch].to(self.device)
        self.real_B = input['cond' if AtoB else 'image'][:divid_batch].to(self.device)

        if if_2:
            self.real_A2 = input['image' if AtoB else 'cond'][divid_batch:].to(self.device)
            self.real_B2 = input['cond' if AtoB else 'image'][divid_batch:].to(self.device)
        # if input2 is not None:
        #     self.real_A2 = input2['A' if AtoB else 'B'].to(self.device)
        #     self.real_B2 = input2['B' if AtoB else 'A'].to(self.device)
        
        # # 用于base_model.get_image_paths之后是sb_test.py【暂废弃】
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if if_scale and not self.opt.use_vqgan:
            self.scale, self.scale_cond = calculate_scale_factor(self.real_A, self.real_B, self.autoencoder, self.device, use_vqgan=self.opt.use_vqgan)

        # with torch.amp.autocast("cuda"):
        self.real_A = apply_autoencoder( self.autoencoder, self.real_A, use_vqgan=self.opt.use_vqgan, recon=False)
        # [b, 1, 160, 160, z]  -> [b, 4, 40, 40, z]
        self.real_B = apply_autoencoder( self.autoencoder, self.real_B, use_vqgan=self.opt.use_vqgan, recon=False)
        self.real_A = self.real_A * self.scale
        self.real_B = self.real_B * self.scale_cond
        if if_2:
            self.real_A2 = apply_autoencoder( self.autoencoder, self.real_A2, use_vqgan=self.opt.use_vqgan, recon=False)
            self.real_B2 = apply_autoencoder( self.autoencoder, self.real_B2, use_vqgan=self.opt.use_vqgan, recon=False)
            self.real_A2 = self.real_A2 * self.scale
            self.real_B2 = self.real_B2 * self.scale_cond


        # 【weights-wjw 251209】
        self.label = input["lb"].to(self.device)
        self.label_tgt = input["lb_tgt"].to(self.device)
        if self.weighted_loss > 1.0:
            # weights: [b, 4, 40, 40, 8]
            weights = torch.ones_like(self.real_A).to(self.device)
            # roi: torch.Size([b, 1, 40, 40, 8])
            roi = torch.zeros([self.real_A.shape[0]] + [1] + list(self.real_A.shape[2:])).to(self.device)
            # 将label: [b, 3, 160, 160, 32]下采样到和inputs一样的size [b, 3, 40, 40, 8]
            interpolate_label = F.interpolate(self.label[:divid_batch], size=list(self.real_A.shape[2:]), mode="nearest")
            for label in self.weighted_loss_label: # label = 0
                # roi[interpolate_label == label] = 1
                roi[(interpolate_label == label).any(dim=1, keepdim=True)] = 1
            weights[roi.repeat(1, self.real_A.shape[1], 1, 1, 1) == 1] = self.weighted_loss
            '''
            # 将weightmap: [b, 3, 160, 160, z] -> [b, 3, 40, 40, z]
            weightmap_d = F.interpolate(weightmap, size=inputs.shape[2:], mode="nearest").to(inputs.device)
            # weights: [b, 4, 40, 40, z]
            weights = torch.cat([weightmap_d, weightmap_d[:, :1]], dim=1).to(inputs.device)
            '''
        else:
            weights = torch.ones_like(self.real_A).to(self.device)
        self.weights = weights
    # forward阶段只用到了netG
    def forward(self):
        
        tau = self.opt.tau
        T = self.opt.num_timesteps # 5
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1),times])
        times = torch.tensor(times).float().cuda()
        # times = [0,1/T,2/T,...,(T-1)/T,1]
        self.times = times
        bs =  self.real_A.size(0) # 1

        # 随机选择时间点
        # (t)time_idx = {0,1,2,...,T-1}
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
        self.time_idx = time_idx # [1]固定是1，不然后续报错
        # self.timestep = times[time_idx]
        
        with torch.no_grad():
            self.netG.eval() #### -----【注意】
            # if self.opt.use_cn:
            #     self.netCN.eval()
            for t in range(self.time_idx.int().item()+1): # 从 0 到 time_idx，一步步加噪
                
                if t > 0:
                    delta = times[t] - times[t-1] # 1/T
                    denom = times[-1] - times[t-1] # 1-(t-1)/T
                    inter = (delta / denom).reshape(-1,1,1,1) # 1/(T-t+1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1) # (T-t)/[(T-t+1)*T]
                # real_A: (1, 1, h, w)
                # Xt: noisy input (step-by-step)
                # z: noise
                Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() \
                    + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                # time_idx: (1, 1)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                # z: (1, 4*64)
                # 【这里是t和z都只在编码后的特征时进行，没有每一步。和maisi有出入，打算先不使用z】
                # 也许可以在maisi里用context=z，但是是每一步的操作
                # #####【看一下论文公式】
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                '''
                Xt: torch.Size([2, 4, 40, 40, 18])
                time_idx: torch.Size([2])
                time_idx: tensor([0, 0], device='cuda:0')
                z: torch.Size([2, 256])
                '''
                # 【目前是把z作为context,但G本身是没有用这个的】
                # <class 'torch.nn.parallel.data_parallel.DataParallel'>
                # Model is wrapped with DataParallel.
                # Number of GPUs used: 8

                # if self.opt.use_cn:
                #     down_block_res_samples, mid_block_res_sample = self.netCN(
                #         x=Xt, timesteps=time_idx, controlnet_cond=self.label[:bs]
                #     )
                #     Xt_1 = self.netG(
                #         x=Xt,
                #         timesteps=time_idx,
                #         context=z,
                #         down_block_additional_residuals=down_block_res_samples,
                #         mid_block_additional_residual=mid_block_res_sample,
                #     )
                # else:
                if self.opt.use_cn:
                    Xt_1     = self.netG(Xt, time_idx, z = z, cond = self.label[:bs])
                else: # input, t, z=None, cond=None
                    Xt_1     = self.netG(Xt, time_idx, context=z)
                # Xt_1: torch.Size([2, 4, 40, 40, 18])

                # real_A, real_A2 (source domain) 相当于batchsize=2?
                # 都是随机加噪的过程
                Xt2       = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()

                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                
                # if self.opt.use_cn:
                #     down_block_res_samples, mid_block_res_sample = self.netCN(
                #         x=Xt2, timesteps=time_idx, controlnet_cond=self.label[bs:]
                #     )
                #     Xt_12 = self.netG(
                #         x=Xt2,
                #         timesteps=time_idx,
                #         context=z,
                #         down_block_additional_residuals=down_block_res_samples,
                #         mid_block_additional_residual=mid_block_res_sample,
                #     )
                # else:
                if self.opt.use_cn:
                    Xt_12     = self.netG(Xt2, time_idx, z = z, cond = self.label[bs:])
                else:
                    Xt_12    = self.netG(Xt2, time_idx, context=z)
                
                if self.opt.nce_idt or self.opt.idt:
                    XtB = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    
                    # 这里没有B的label所以不用cn了
                    if self.opt.use_cn:
                        Xt_1B = self.netG(XtB, time_idx, z=z, cond = self.label_tgt[:bs])
                    else:
                        Xt_1B = self.netG(XtB, time_idx, context=z)
            if self.opt.nce_idt or self.opt.idt:
                self.XtB = XtB.detach()
            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()
                      
        z_in    = torch.randn(size=[2*bs,4*self.opt.ngf]).to(self.real_A.device)
        z_in2    = torch.randn(size=[bs,4*self.opt.ngf]).to(self.real_A.device)
        """Run forward pass"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if (self.opt.nce_idt or self.opt.idt) and self.opt.isTrain else self.real_A
        
        self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if (self.opt.nce_idt or self.opt.idt) and self.opt.isTrain else self.real_A_noisy
        
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                self.realt = torch.flip(self.realt, [3])
        
        # [At, Bt] -> fake=[fake_B, idt_B]
        # At_2 -> fake_B2
        # with torch.amp.autocast("cuda"):
        

        # if self.opt.use_cn:
        #     down_block_res_samples, mid_block_res_sample = self.controlnet(
        #         x=self.realt, timesteps=self.time_idx, controlnet_cond= torch.cat((self.label[:bs], self.label_tgt[:bs]), dim=0) 
        #     )
        #     self.fake = self.netG(
        #         x=self.realt,
        #         timesteps=self.time_idx,
        #         context=z_in,
        #         down_block_additional_residuals=down_block_res_samples,
        #         mid_block_additional_residual=mid_block_res_sample,
        #     )

        #     down_block_res_samples, mid_block_res_sample = self.controlnet(
        #         x=self.real_A_noisy2, timesteps=self.time_idx, controlnet_cond=self.label[bs:]
        #     )
        #     self.fake_B2 = self.netG(
        #         x=self.real_A_noisy2,
        #         timesteps=self.time_idx,
        #         context=z_in2,
        #         down_block_additional_residuals=down_block_res_samples,
        #         mid_block_additional_residual=mid_block_res_sample,
        #     )
        # else:
        if self.opt.use_cn:
            self.fake     = self.netG(self.realt, self.time_idx, z = z_in, cond = torch.cat((self.label[:bs], self.label_tgt[:bs]), dim=0))
            self.fake_B2     = self.netG(self.real_A_noisy2, self.time_idx, z = z_in2, cond = self.label[bs:])
        else:
            self.fake = self.netG(self.realt,self.time_idx,context=z_in)
            self.fake_B2 =  self.netG(self.real_A_noisy2,self.time_idx,context=z_in2)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt or self.opt.idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        if self.opt.phase == 'test': # 未改
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1),times])
            times = torch.tensor(times).float().cuda()
            self.times = times
            bs =  self.real.size(0) # 【time_idx可能存在问题！！！】
            time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
            self.time_idx = time_idx
            self.timestep = times[time_idx]
            visuals = []
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):
                    
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1,1,1,1) # s(t_j+1)
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1) # s(t_j+1) * (1 - s(t_j+1)) * (1 - t_j)
                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1     = self.netG(Xt, time_idx, context=z)
                    
                    setattr(self, "fake_"+str(t+1), Xt_1)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        bs =  self.real_A.size(0)
        
        fake = self.fake_B.detach() # G(At) = fake_B
        std = torch.rand(size=[1]).item() * self.opt.std
        
        pred_fake = self.netD(fake,self.time_idx) # 结果是list,可能需要修改loss
        self.pred_real = self.netD(self.real_B,self.time_idx)
        
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean() # 应该检测出fake
        loss_D_real = self.criterionGAN(self.pred_real, True) # 应该为True的标准
        self.loss_D_real = loss_D_real.mean()
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # if dist.is_initialized():
        #     dist.all_reduce(self.loss_D, op=torch.distributed.ReduceOp.SUM)
            # world_size = dist.get_world_size() 
            # self.loss_D = self.loss_D / world_size
        return self.loss_D
    def compute_E_loss(self):
        
        bs =  self.real_A.size(0)
        
        """Calculate GAN loss for the discriminator"""
        
        # At,G(At)
        XtXt_1 = torch.cat([self.real_A_noisy,self.fake_B.detach()], dim=1)
        # At_2,G(At_2)
        XtXt_2 = torch.cat([self.real_A_noisy2,self.fake_B2.detach()], dim=1)

        # netE(input1, t, input2) 会把两个input拼接输入
        # 拉伸成一维，然后取logsumexp，再取mean  相当于衡量了差异 pred_fake
        # 减去  logsumexp(netE(input1, t, input1))  相当于衡量了一致性 pred_real
        
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() +temp + temp**2
        # if dist.is_initialized():
        #     dist.all_reduce(self.loss_E, op=torch.distributed.ReduceOp.SUM)
            # world_size = dist.get_world_size() 
            # self.loss_E = self.loss_E / world_size
        return self.loss_E
    def compute_G_loss(self):
        
        bs =  self.real_A.size(0)
        tau = self.opt.tau # tau=0.01
        
        """Calculate GAN and NCE loss for the generator"""

        fake = self.fake_B # G(At) = fake_B 本来没有detach()
        std = torch.rand(size=[1]).item() * self.opt.std
        # print('fake:',fake.shape) # [2, 4, 40, 40, 18]
        # fake = fake[:,:,:,:,0]
        # print('fake:',fake.shape)
        if self.opt.lambda_GAN > 0.0: # 1.0
            pred_fake = self.netD(fake,self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0: # 1.0
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            
            # bs = self.opt.batch_size
            ET_XY    = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            self.loss_SB = -(self.opt.num_timesteps-self.time_idx[0])/self.opt.num_timesteps*self.opt.tau*ET_XY
            mse=(((self.real_A_noisy-self.fake_B)**2)* self.weights).mean()
            # torch.mean((self.real_A_noisy-self.fake_B)**2)
            self.loss_SB += self.opt.tau*mse
        
        torch.cuda.empty_cache()
        if self.opt.lambda_NCE > 0.0: # 1.0
            label = torch.cat((self.label[:bs], self.label_tgt[:bs]), dim=0) # 
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake, y=label.detach()) # feature-level
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0: # no
            label = torch.cat((self.label_tgt[:bs], self.label_tgt[:bs]), dim=0)
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B,y=label.detach())
            # self.loss_NCE_Y = 0.0
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        

        self.loss_G = self.loss_G_GAN + self.opt.lambda_SB*self.loss_SB + self.opt.lambda_NCE*loss_NCE_both
        
        if self.opt.idt and self.opt.lambda_idt > 0.0:
            self.loss_idt = self.criterionIdt(self.real_B, self.idt_B) * self.opt.lambda_idt
            self.loss_G = self.loss_G + self.loss_idt
        else:
            self.loss_idt = 0.0
        
        # if dist.is_initialized():
        #     dist.all_reduce(self.loss_G, op=torch.distributed.ReduceOp.SUM)
            # world_size = dist.get_world_size() 
            # self.loss_G = self.loss_G / world_size
        return self.loss_G


    # img->G->F = feature
    # 需大改netG
    def calculate_NCE_loss(self, src, tgt, y=None):
        n_layers = len(self.nce_layers) # [0,4,8,12,16] 【注意长度】
        # 【注意z改了】
        # z    = torch.randn(size=[self.real_A.size(0),4*self.opt.ngf]).to(self.real_A.device)
        z    = torch.randn(size=[2*self.real_A.size(0),4*self.opt.ngf]).to(self.real_A.device)
        ## 【注意netG和netF】
        
        # # 原本的在DDP框架下不行，多次调用netG不行
        # feat_q = self.netG(tgt, self.time_idx*0, context=z, layers=self.nce_layers, encode_only=True)
        # feat_k = self.netG(src, self.time_idx*0, context=z,layers=self.nce_layers, encode_only=True)
        bs = src.size(0)
        # with torch.amp.autocast("cuda"):
        input = torch.cat([tgt, src], dim=0)
        
        
        if self.opt.use_cn:
            # down_block_res_samples, mid_block_res_sample = self.netCN(
            #     x=input.clone().detach().requires_grad_(True), timesteps=torch.zeros_like(self.time_idx), controlnet_cond=y.detach()
            # )
            # down_block_res_samples = [t.clone() for t in down_block_res_samples]
            # mid_block_res_sample = mid_block_res_sample.clone()
            # feat = self.netG(
            #     x=input.clone().detach().requires_grad_(True),
            #     timesteps=torch.zeros_like(self.time_idx),
            #     context=z,
            #     down_block_additional_residuals=down_block_res_samples,
            #     mid_block_additional_residual=mid_block_res_sample,
            #     layers=self.nce_layers, encode_only=True
            # )
            # del down_block_res_samples, mid_block_res_sample
            feat = self.netG(input, self.time_idx*0, z=z, cond = y, layers=self.nce_layers, encode_only=True)
        else:
            feat = self.netG(input, self.time_idx*0, context=z, layers=self.nce_layers, encode_only=True)
        feat_q, feat_k = [], []
        for feat_i in feat:
            feat_q.append(feat_i[:bs])
            feat_k.append(feat_i[bs:])
        '''
        以2D为例，输入[2, 4, 46, 46] -> [2, 46*46, 4] --sample--> [2, 256, 4]
        -> [512, 4] --MLP--> [512, 256] 对应id=[256]       
        '''
        if self.opt.flip_equivariance and self.flipped_for_equivariance: # no
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        
        # self.opt.num_patches=256
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()
        # if dist.is_initialized():
        #     dist.all_reduce(total_nce_loss, op=torch.distributed.ReduceOp.SUM)
        return total_nce_loss / n_layers

    def get_current_visuals(self):
        from collections import OrderedDict
        from scripts.sample import ReconModel
        # self.real_A: torch.Size([1, 4, 40, 40, 18])
        if not self.opt.use_vqgan:
            recon_model = ReconModel(autoencoder=self.autoencoder, scale_factor=self.scale).to(self.device)
    
        real_A = apply_autoencoder(recon_model if not self.opt.use_vqgan else self.autoencoder, 
                                             self.real_A[:1], use_vqgan=self.opt.use_vqgan, recon=True,scale_factor=self.scale)
        
        real_B = apply_autoencoder(recon_model if not self.opt.use_vqgan else self.autoencoder, 
                                             self.real_B[:1], use_vqgan=self.opt.use_vqgan, recon=True,scale_factor=self.scale)
        
        fake_B = apply_autoencoder(recon_model if not self.opt.use_vqgan else self.autoencoder, 
                                             self.fake_B[:1], use_vqgan=self.opt.use_vqgan, recon=True,scale_factor=self.scale)
        
        if self.opt.nce_idt or self.opt.idt:
            idt_B = apply_autoencoder(recon_model if not self.opt.use_vqgan else self.autoencoder, 
                                             self.idt_B[:1], use_vqgan=self.opt.use_vqgan, recon=True,scale_factor=self.scale)

            return OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                            ('real_B', real_B),
                            ('idt_B', idt_B)])#
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                            ('real_B', real_B)])