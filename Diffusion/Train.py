
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], 
                     ch=modelConfig["channel"], 
                     ch_mult=modelConfig["channel_mult"], 
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], 
                     dropout=modelConfig["dropout"] ).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    
    # 优化器
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    
    # 学习率策略 https://blog.csdn.net/weixin_44682222/article/details/122218046
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    
    # 学习率预热
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    
    # 训练器
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                
                # 前向传播，计算梯度
                loss = trainer(x_0).sum() / 1000. # 这里为什么要除以 1000 呢？
                
                # 反向传播计算梯度
                loss.backward()
                
                # 梯度裁剪（通过设定阈值来解决梯度消失或梯度爆炸的问题）https://blog.csdn.net/Mikeyboi/article/details/119522689
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip_max_norm"])
                
                # 参数更新
                optimizer.step()
                
                # log msg output
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        
        # 没明白是怎么回事
        warmUpScheduler.step() 
        
        # 保存 check_point
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], 
                     ch=modelConfig["channel"], 
                     ch_mult=modelConfig["channel_mult"], 
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], 
                     dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], 
            modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        
        # 将模型设置为 eval() 模式
        model.eval()
        
        sampler = GaussianDiffusionSampler(
            model, 
            modelConfig["beta_1"], 
            modelConfig["beta_T"], 
            modelConfig["T"]).to(device)
        
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], # batch_szie, channel, H, W
            device=device) # [-1 ~ 1]
        
        # 将 noisyImage 以可视化的形式保存下来
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(
            saveNoisy, 
            os.path.join(modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), 
            nrow=modelConfig["nrow"])
        
        # 将 noisyImage 输入到模型中进行计算和采样
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [-1 ~ 1] => [0 ~ 1]
        
        # 将采样得到的 image 保存下来
        save_image(
            sampledImgs, 
            os.path.join(modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), 
            nrow=modelConfig["nrow"])