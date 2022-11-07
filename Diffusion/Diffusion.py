
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device) # https://zhuanlan.zhihu.com/p/352877584
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        
        # https://blog.csdn.net/weixin_38145317/article/details/104917218
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0) # y_{i} = x_{1} * x_{2} * … * x_{i} , 注意，输出维度与输入维度相同。
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T] # F.pad: tensor扩充函数, 扩充完之后，截取前面 T 个元素。https://blog.csdn.net/jorg_zhao/article/details/105295686

        # 下面两个是 \tilde{\mu}_t 式子的两个系数 （当然，这里是该式子的一系列系数）
        self.register_buffer('coeff1', torch.sqrt(1. / alphas)) # 将一些常量储存在 buffer 里面。  
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # 下面这个是 \tilde{\beta}_t 的值 （当然，这里是一系列的该值）
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        
        # for test
        self.register_buffer('var', torch.cat([self.posterior_var[1:2], self.betas[1:]]) )


    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t): # 下面的 posterior_var 好像就只是用到了 idx=1 的元素，其他元素都没用到诶
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]]) # 这里为什么这样 concat 呢？ 这里为什么不用 buffer 来代替呢？作者给的回复是，这里的 var 和 betas 类似，即使代替也不影响
        
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps) # 预测的 x_{t-1} 的均值，在这里是直接用公式计算
        
        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)): # reversed 返回一个反转的迭代器
            print(time_step)
            
            # 返回一个与size大小相同的，且用 1 填充的张量，这里实际是 batch_size 个 time_step 组成的一个数组。
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            
            # 这里的 mean 指的是预测的 x_{t-1} 的均值，而 var 其实就是之前设定的 x_{t-1} 的 方差
            mean, var= self.p_mean_variance(x_t=x_t, t=t) 
            
            # no noise when time_step == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            
            x_t = mean + torch.sqrt(var) * noise
            
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


