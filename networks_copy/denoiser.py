import torch
import torch.nn as nn
import torch.nn.functional as F
# from model_jit import JiT_models
from scripts.utils import define_instance

def judge_requires_grad(obj):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).requires_grad
    else:
        raise TypeError
    
class RequiresGradContext(object):
    def __init__(self, *objs, requires_grad):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]
        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError
        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

def cosine_similarity(X,Y):
    '''
    compute cosine similarity for each pair of image
    Input shape: (batch,channel,H,W)
    Output shape: (batch,1)
    '''
    X_flat = X.reshape(X.shape[0], X.shape[1], -1)
    Y_flat = Y.reshape(Y.shape[0], Y.shape[1], -1)

    corr = norm(X_flat) * norm(Y_flat)
    similarity = corr.sum(dim=1).mean(dim=1)

    return similarity

def mse(x,y):
    return (x - y).square().sum(dim=tuple(range(1, x.dim())))

class Denoiser(nn.Module):
    def __init__(
        self,
        args,
        device,
        debug=False
    ):
        super().__init__()

        self.net = define_instance(args, "diffusion_unet_def").to(device)
        self.device = device

        # JiT:(-0.8,0.8) # 但是是它单独的经验调参
        self.P_mean = args.flow_matching_train["P_mean"]
        self.P_std = args.flow_matching_train["P_std"] # 0.8
        self.t_eps = args.flow_matching_train["t_eps"] # 5e-2
        self.noise_scale = args.flow_matching_train["noise_scale"] # 1.0
        if args.mode == "train_cn":
            self.weighted_loss = args.controlnet_train["weighted_loss"]
        else:
            self.weighted_loss = args.flow_matching_train["weighted_loss"]

        # ema 保存两个太占内存了
        self.ema_decay1 = args.flow_matching_train["ema_decay1"] # 0.9999
        self.ema_params1 = None
        # self.ema_decay2 = getattr(args, name)["ema_decay2"] # 0.9996
        # self.ema_params2 = None

        # generation hyper params
        
        self.method = args.flow_matching_sampling["sampling_method"] # heun
        
        self.cfg_interval = args.flow_matching_sampling["interval"]

        if debug: # 参数都由py文件的args传入，非json调参
            self.latent_shape = tuple(args.latent_shape)
            self.cfg_scale = args.cfg
            self.steps = args.steps
        else:
            self.latent_shape = None
            self.cfg_scale = args.flow_matching_sampling["cfg"]
            self.steps = args.flow_matching_sampling["num_sampling_steps"] # 50



    def sample_t(self, n: int, device=None):
        # N(0,1) => N(mean, std)
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z) # 映射回(0,1)区间

    def forward(self, x, controlnet, labels = None):
        self.latent_shape = x.shape[1:]
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e # z_t 加噪图像
        v = (x - z) / (1 - t).clamp_min(self.t_eps) # 防止零除问题

        # 模型直接预测输出是x（x-预测），然后转化为v
        if controlnet:
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=z, timesteps=t.flatten(), controlnet_cond=labels
            )
            x_pred = self.net(
                x=z, 
                timesteps=t.flatten(),
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
        else:
            x_pred = self.net(x=z, timesteps=t.flatten())



        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        
        if self.weighted_loss > 1.0: # l2 loss 
            # loss = (v - v_pred) ** 2
            loss = torch.abs(v - v_pred)

            weighted_loss_label = [0]
            weights = torch.ones_like(x).to(x.device)
            roi = torch.zeros([x.shape[0]] + [1] + list(x.shape)[2:]).to(x.device)
            interpolate_label = F.interpolate(labels, size=x.shape[2:], mode="nearest")
            for label in weighted_loss_label:
                roi[(interpolate_label == label).any(dim=1, keepdim=True)] = 1
            weights[roi.repeat(1, x.shape[1], 1, 1, 1) == 1] = self.weighted_loss
            # [b, latent, x, y, z]
            loss = (loss * weights).mean()
        else: # LOSS nan, modify to L1
            # loss = loss.mean()
            loss_pt = torch.nn.L1Loss()
            loss = loss_pt(v_pred.float(), v.float())

        return loss

    @torch.no_grad()
    def generate(self, controlnet = None, labels = None, x=None, t_start=0.25):

        if x is None:
            z = self.noise_scale * torch.randn((1,)+self.latent_shape, device=self.device)
            timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=self.device).view(-1, *([1] * z.ndim)).expand(-1, 1, -1, -1, -1, -1)
        else:
            e = torch.randn_like(x) * self.noise_scale
            timesteps = torch.linspace(t_start, 1.0, self.steps+1, device=self.device).view(-1, *([1] * e.ndim)).expand(-1, 1, -1, -1, -1, -1)
            z = t_start * x + (1 - t_start) * e

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun": # 这个
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1): # 50-1
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, controlnet, labels)
        # last step euler # 最后一步固定用euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], controlnet, labels)
        return z
    
    # @torch.no_grad()
    def translation(self, controlnet, labels, src, tgt=None, t_start=0.25, lambda_list=[0,0]):

        if src is None:
            z = self.noise_scale * torch.randn((1,)+self.latent_shape, device=self.device)
            timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=self.device).view(-1, *([1] * z.ndim)).expand(-1, 1, -1, -1, -1, -1)
        else:
            e = torch.randn_like(src) * self.noise_scale
            timesteps = torch.linspace(t_start, 1.0, self.steps+1, device=self.device).view(-1, *([1] * e.ndim)).expand(-1, 1, -1, -1, -1, -1)
            z = t_start * src + (1 - t_start) * e

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun": # 这个
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1): # 50-1
            t = timesteps[i]
            t_next = timesteps[i + 1]
            x_ref = t * src + (1 - t) * e
            z = stepper(z, t, t_next, controlnet, labels, lambda_list=lambda_list, x_ref=x_ref)
        # last step euler # 最后一步固定用euler
        x_ref = timesteps[-2] * src + (1 - timesteps[-2]) * e
        z = self._euler_step(z, timesteps[-2], timesteps[-1], controlnet, labels, lambda_list=lambda_list,x_ref=x_ref)


        return z.detach()

    def compute_energy(self, x_pred, x_ref, lambda_list):
        """
        x_pred: 当前模型预测出的无噪声 3D latent (B, 4, 40, 40, 18)
        x_ref:  源图像的无噪声 3D latent
        lambda_list: [w_style, w_struct]，例如 [-0.1, 0.5]
        """
        w_style, w_struct = lambda_list
        
        # 1. DSE (风格/语义特征引导)
        # 目标: 远离源域 (w_style 通常为负数，或者计算负相关)
        if w_style != 0.0:
            # 简单示例: 直接比较特征空间的 Cosine 相似度或 L2
            # 相似度越高，能量越大 (如果 w_style 为负，则模型会尽量降低相似度)
            # F.mse_loss(x_pred_low, x_ref_low, reduction='sum')
            style_loss = cosine_similarity(x_pred, x_ref)
        else:
            style_loss = 0.0
            
        # 2. DIE (结构/低频特征引导)
        # 目标: 靠近源域结构 (w_struct 通常为正数)
        if w_struct != 0.0:
            # 使用 3D 平均池化抹除高频细节，提取宏观拓扑结构
            pool_kernel = 3
            x_pred_low = F.avg_pool3d(x_pred, kernel_size=pool_kernel, stride=1, padding=pool_kernel//2)
            x_ref_low  = F.avg_pool3d(x_ref,  kernel_size=pool_kernel, stride=1, padding=pool_kernel//2)
            # print('befor pool:',x_pred.shape)
            # print('pool shape:',x_pred_low.shape) # [1, 4, 40, 40, 18]
            struct_loss = mse(x_pred_low, x_ref_low)
        else:
            struct_loss = 0.0
        return style_loss,struct_loss
    

    # @torch.no_grad()
    def _forward_sample(self, z, t, controlnet, labels, lambda_list=[0,0], x_ref=None):

        # unconditional
        with torch.no_grad():
            x_uncond = self.net(z, t.flatten())
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        if controlnet is None:
            return v_uncond


        

        # 2. 能量梯度引导，【局部开启梯度】
        lambda_style, lambda_struct = lambda_list
        if (lambda_style!=0.0 or lambda_struct != 0.0 ) and x_ref is not None:
            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=z, timesteps=t.flatten(), controlnet_cond=labels
                )
            
            with RequiresGradContext(z, requires_grad=True):
                # 让 z 能够计算梯度
                # z_in = z.detach().requires_grad_(True)

                x_cond = self.net(
                    x=z, 
                    timesteps=t.flatten(),
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
                
                # Joint energy gradient: differentiate the shared graph once.
                style_energy, structure_energy = self.compute_energy(x_cond, x_ref, lambda_list)
                energy_total = 0.0
                if lambda_style != 0.0:
                    energy_total = energy_total + lambda_style * style_energy.sum()
                if lambda_struct != 0.0:
                    energy_total = energy_total + lambda_struct * structure_energy.sum()
                grad_total = torch.autograd.grad(energy_total, z)[0]
        else:
            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=z, timesteps=t.flatten(), controlnet_cond=labels
                )
                x_cond = self.net(
                    x=z, 
                    timesteps=t.flatten(),
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
            
        
        
        v_cond = (x_cond.detach() - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval # 0.1, 1.0
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0) # 2.9

        # 如果t在区间内，则1.0，为v_cond;
        # t 不在区间内，则使用args.cfg = 2.9(3.0), 增强
        v_pred = v_uncond + cfg_scale_interval * (v_cond - v_uncond)
        
        # Preserve the original weighted-energy update, with a single backward pass.
        if (lambda_style != 0.0 or lambda_struct != 0.0) and x_ref is not None:
            v_pred = v_pred - t * grad_total.detach()

        return v_pred.detach()

    @torch.no_grad()
    def _forward_sample_raw_bug(self, z, t, controlnet, labels, lambda_list=[0,0], x_ref=None):

        # unconditional
        x_uncond = self.net(z, t.flatten())
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        if controlnet is None:
            return v_uncond


        down_block_res_samples, mid_block_res_sample = controlnet(
            x=z, timesteps=t.flatten(), controlnet_cond=labels
        )
        x_cond = self.net(
            x=z, 
            timesteps=t.flatten(),
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval # 0.1, 1.0
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0) # 2.9

        # 如果t在区间内，则1.0，为v_cond;
        # t 不在区间内，则使用args.cfg = 2.9(3.0), 增强
        v_pred = v_uncond + cfg_scale_interval * (v_cond - v_uncond)

        # 2. 能量梯度引导，【局部开启梯度】
        lambda_style, lambda_struct = lambda_list
        if (lambda_style!=0.0 or lambda_struct != 0.0 ) and x_ref is not None:
            with torch.enable_grad():
                # 让 z 能够计算梯度
                z_in = z.detach().requires_grad_(True)
                
                # 计算能量 【在此修改】
                energy1,energy2 = self.compute_energy(x_cond, x_ref, lambda_list) 
                
                # 求导: d(Energy) / d(z)
                if lambda_style !=0.0:
                    grad1 = torch.autograd.grad(energy1, z_in)[0].detach()
                else:
                    grad1 = 0.0
                if lambda_struct !=0.0:
                    grad2 = torch.autograd.grad(energy2, z_in)[0].detach()
                else:
                    grad2 = 0.0
                
            # 利用梯度修正预测的速度 v
            # weight_t 可以根据时间步 t 动态调整，比如初期引导强，后期引导弱
            # weight_t = t # 类似 DDPM 中的衰减
            if lambda_style !=0.0:
                v_pred = v_pred - lambda_style * t * grad1
            if lambda_struct !=0.0:
                v_pred = v_pred - lambda_struct * t *grad2
        return v_pred

    # @torch.no_grad()
    def _euler_step(self, z, t, t_next, controlnet, labels, lambda_list=[0,0], x_ref=None):
        # 欧拉：一阶数值积分
        # 
        v_pred = self._forward_sample(z, t, controlnet, labels,lambda_list=lambda_list, x_ref=x_ref)
        z_next = z + (t_next - t) * v_pred
        return z_next

    # @torch.no_grad()
    def _heun_step(self, z, t, t_next, controlnet, labels,lambda_list=[0,0], x_ref=None):
        # 二阶
        v_pred_t = self._forward_sample(z, t, controlnet, labels,lambda_list=lambda_list, x_ref=x_ref)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, controlnet, labels,lambda_list=lambda_list, x_ref=x_ref)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        # 从当前模型参数 self.parameters() 中取出最新的参数 src
        # 按照指数滑动平均公式更新 EMA 参数 targ
        # targ <- targ×ema_decay1 + src×(1−ema_decay1)
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        # for targ, src in zip(self.ema_params2, source_params):
        #     targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
