import torch
import torch.nn as nn
import numpy as np
from functools import partial
from positional_encoding import SinusoidalPositionEmbeddings, DiffusionEmbedding
from inspect import isfunction
import torch.nn.functional as F
import math
from typing import List, Callable

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):  # 目的是沿着指定的索引张量 t 从张量 a 中提取值，并根据 x_shape 的形状重新整形结果。
    b, *_ = t.shape
    t = t.long()
    out = a.gather(-1, t)  # 使用 gather 方法从张量 a 的最后一个维度（-1）提取使用索引张量 t 指定的值。该操作实际上是沿着 a 的最后一个维度使用 t 中的值进行索引。结果存储在变量 out 中。
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DiffusionTrainingNetwork(nn.Module):
    def __init__(self, args, num_nodes, beta_end=0.1, beta_schedule='linear', num_layers=2, dropout_rate=0.1):
        super(DiffusionTrainingNetwork, self).__init__()
        self.cond_linear = nn.Linear(args.n_hidden*2, args.n_hidden)
        self.num_nodes = num_nodes
        self.t_enc = SinusoidalPositionEmbeddings(args.n_hidden)

        self.denoise_fn = EpsilonTheta(args, self.num_nodes, args.input_dropout, args.hidden_dropout, args.feat_dropout)
        # self.denoise_fn = EpsilonTheta_New(args)
        self.diffusion = GaussianDiffusion(self.denoise_fn, args, beta_end=beta_end, beta_schedule=beta_schedule)
    
    def forward(self, seq_embs, time_index):
        # (batch_size, sub_seq_len, input_dim)
        # time_embs = torch.IntTensor([time_index]).to(seq_embs.device)#.reshape(-1, 1)
        # time_embs = self.t_enc(time_embs)
        # time_embs = time_embs.repeat_interleave(repeats=seq_embs.size(0), dim=0)
        # inputs = torch.cat((seq_embs, time_embs), dim=-1)
        # cond = self.cond_linear(inputs)
        return seq_embs, time_index  # cond
    
    def train_diffusion(self, history_embs, target_embs, time_index, ent_embeddings):
        cond, time_emb = self.forward(history_embs, time_index)
        likelihoods = self.diffusion.log_prob(target_embs, cond, ent_embeddings, time_emb)
        return likelihoods
    
    def sampling_decoder(self, seq_embs, time_index, ent_embeddings):
        cond, time_emb = self.forward(seq_embs, time_index)
        new_samples = self.diffusion.sample(cond=cond.unsqueeze(1), ent_embeddings=ent_embeddings, time_emb=time_emb)
        return new_samples

class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, args, beta_end=0.1, beta_schedule='linear'):
        super().__init__()
        self.args = args
        self.denoise_fn = denoise_fn
        self.diff_steps = args.steps
        if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, self.diff_steps)  # diff_steps: 扩散过程的步数，默认为 100 步。
        elif beta_schedule == "quad":
            betas = np.linspace(1e-4 ** 0.5, beta_end ** 0.5, self.diff_steps) ** 2
        elif beta_schedule == "const":
            betas = beta_end * np.ones(self.diff_steps)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(self.diff_steps, 1, self.diff_steps)
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, self.diff_steps)
            betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.diff_steps)
        else:
            raise NotImplementedError(beta_schedule)
        
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)  # 计算 alphas 的累积乘积。
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])  # 创建 alphas_cumprod 的前一个版本，首位添加一个 1.0。

        to_torch = partial(torch.tensor, dtype=torch.float32)  # 创建一个局部函数 to_torch，用于将 numpy 数组转换为 torch 张量。

        self.register_buffer("betas", to_torch(betas))  # register_buffer 方法用于将一个张量注册为模型的缓冲区（buffer）。缓冲区是模型中的一种特殊张量，其数值不会被更新（即不会被优化器修改），但会被保存和加载。通常，缓冲区用于存储模型的固定参数、统计信息、移动平均值等。
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (  # 计算了后验分布的方差
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(  # 计算了后验分布的均值的系数
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def log_prob(self, x, cond, ent_embeddings, time_emb):
        B, _ = x.shape
        steps = torch.randint(0, self.diff_steps, (B,), device=x.device).long()
        loss = self.p_losses(x.reshape(B, 1, -1), cond.reshape(B, 1, -1), steps, ent_embeddings, time_emb)
        return loss
    
    def p_losses(self, x_start, cond, steps, ent_embeddings, time_emb, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))  # 如果没有提供噪声 (noise)，则使用 torch.randn_like 生成与 x_start 相同形状的随机噪声。
        x_noisy = self.q_sample(x_start=x_start, steps=steps, noise=noise)  # 使用 q_sample 方法生成扩散链的初始样本。这是通过对初始样本应用扩散过程的逆过程来实现的。在这里，x_noisy 表示带有噪声的初始样本。
        noise_predict = self.denoise_fn(x_noisy, steps, ent_embeddings, cond, time_emb)  # 使用给定的去噪函数 (denoise_fn) 对带噪声的初始样本进行去噪，以得到 x_recon。去噪的目的是还原观测数据中的真实信号，去除由扰动引入的噪声。
        loss = F.mse_loss(noise_predict, noise)
        return loss
    
    def q_sample(self, x_start, steps, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, steps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, steps, x_start.shape) * noise
        )
    
    # def sample(self, cond, ent_embeddings, time_emb):
    #     b = cond.size(0)
    #     img = torch.randn(cond.size(0), 1, cond.size(-1), device=self.args.device)
    #     for i in reversed(range(0, self.diff_steps)):  # 从扩散过程的最后一步开始向前遍历。
    #         step = i
    #         alpha = self.alphas_cumprod[step]
    #         beta = self.betas[step]
    #         sigma = beta
    #         if step == 0:
    #             z = 0
    #         else:
    #             z = torch.randn(img.shape, device=img.device)
    #         pred_noise = self.denoise_fn(img, torch.full((b,), i, device=self.args.device, dtype=torch.int32), ent_embeddings, cond=cond, time_emb=time_emb)
    #         img = (img - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + sigma.sqrt() * z
    #         # img = self.p_sample(img, cond, ent_embeddings, torch.full((b,), i, device=self.args.device, dtype=torch.int32))
    #     return img
    def sample(self, cond, ent_embeddings, time_emb):
        b = cond.size(0)
        img = torch.randn(cond.size(0), 1, cond.size(-1), device=self.args.device)
        for i in reversed(range(0, self.diff_steps)):  # 从扩散过程的最后一步开始向前遍历。
            img = self.p_sample(img, cond, ent_embeddings, torch.full((b,), i, device=self.args.device, dtype=torch.int32), time_emb)
        return img


    def p_sample(self, x, cond, ent_embeddings, step, time_emb):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, ent_embeddings = ent_embeddings, step=step, time_emb=time_emb)  # 获取模型在当前时刻的均值、方差和对数方差
        noise = torch.randn(x.shape, device=device)  # 生成一个与 x 相同形状的噪声。[700,1,370]
        nonzero_mask = (1 - (step == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))  # 创建一个与 x 形状相同的张量，其中当 t == 0 时为 1，否则为 0。用于在时间步为 0 时不添加噪声。
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise  # 将噪声应用于模型均值，得到采样结果。如果 t == 0，则不添加噪声。
        

    def p_mean_variance(self, x, cond, ent_embeddings, step, time_emb):
        x_recon = self.predict_start_from_noise(
            x, step=step, noise=self.denoise_fn(x, step, ent_embeddings, cond, time_emb)  # 使用 predict_start_from_noise 方法通过去噪函数 denoise_fn 还原初始样本。
        )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, step=step
        )
        return model_mean, posterior_variance, posterior_log_variance
    
    def predict_start_from_noise(self, x_t, step, noise):  # 用于通过噪声还原起始样本。
        return (
            extract(self.sqrt_recip_alphas_cumprod, step, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, step, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, step):  # x_start是去噪以后的，x_t是去噪以前的，t是扩散的步数
        posterior_mean = (
            extract(self.posterior_mean_coef1, step, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, step, x_t.shape) * x_t
        )  # [700,1,370]  后验均值
        posterior_variance = extract(self.posterior_variance, step, x_t.shape)  # 后验方差  [700,1,1]
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, step, x_t.shape  # 后验对数方差  [700,1,1]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

class EpsilonTheta(nn.Module): # 用来预测噪声的模型
    def __init__(self, args, num_nodes, input_dropout=0, hidden_dropout=0, feat_dropout=0, channels=50, kernel_size=5, use_bias=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.i_enc = SinusoidalPositionEmbeddings(args.n_hidden)  # 扩散步数的encoder
        # self.i_enc = DiffusionEmbedding(args.n_hidden, proj_dim=args.n_hidden)  # 使用定义的PositionalEncoding模块创建扩散步数编码器。

        self.inp_drop = torch.nn.Dropout(input_dropout)
        # self.feature_map_drop = torch.nn.Dropout(feat_dropout)
        # self.conv1 = torch.nn.Conv1d(4, channels, kernel_size, stride=1,
        #                        padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        # self.bn0 = torch.nn.BatchNorm1d(4)
        # self.bn1 = torch.nn.BatchNorm1d(channels)
        # self.fc = nn.Linear(args.n_hidden*channels, args.n_hidden)#nn.Linear(args.n_hidden*channels, args.n_hidden)

        self.linear = nn.Linear(args.n_hidden, args.n_hidden)

        self.linear_x = nn.Linear(args.n_hidden, args.n_hidden)
        self.linear_steps = nn.Linear(args.n_hidden, args.n_hidden)
        self.linear_cond = nn.Linear(args.n_hidden, args.n_hidden)
        self.linear_ent_emb = nn.Linear(args.n_hidden, args.n_hidden)
        self.linear_time_emb1 = nn.Linear(args.n_hidden, args.n_hidden)
        self.linear_time_emb2 = nn.Linear(args.n_hidden, args.n_hidden)
        # self.init_weights()

    def init_weights(self):
        # nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.linear_x.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.linear_steps.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.linear_cond.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.linear_ent_emb.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.linear_time_emb1.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.linear_time_emb2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear_x.weight)
        nn.init.xavier_uniform_(self.linear_steps.weight)
        nn.init.xavier_uniform_(self.linear_cond.weight)
        nn.init.xavier_uniform_(self.linear_ent_emb.weight)
        nn.init.xavier_uniform_(self.linear_time_emb1.weight)
        nn.init.xavier_uniform_(self.linear_time_emb2.weight)
        

    def forward(self, x, steps, ent_embeddings, cond, time_emb):
        # shape = x.shape
        # steps = self.i_enc(steps.unsqueeze(1))
        # # x = torch.cat([x, steps, cond], 1)
        # x = torch.cat([x, steps, cond, ent_embeddings.unsqueeze(1)], 1)
        # # x = torch.cat([x, steps, cond+ent_embeddings.unsqueeze(1)], 1)

        # # # nodes_range = torch.arange(0, self.num_nodes, dtype=torch.int32, device=x.device).reshape(self.num_nodes, 1, 1)
        # # # nodes_id_embs = self.id_enc(nodes_range)
        # # steps = self.i_enc(steps.reshape(-1,1,1))
        # # stacked_inputs = torch.cat([x, steps, cond], 1)
        # # # stacked_inputs = self.bn0(stacked_inputs)
        # # x = self.inp_drop(x)
        # x = self.conv1(x)
        # # # x = self.bn1(x)
        # # x = F.leaky_relu(x, 0.4)
        # x = F.relu(x)
        # # x = self.feature_map_drop(x)
        # x = x.view(shape[0], -1)


        # x = self.fc(x)
        # x = x.view(*shape)
        # return x

        shape = x.shape
        steps = self.i_enc(steps.unsqueeze(1))
        # x = torch.cat([x, steps, cond, ent_embeddings.unsqueeze(1)], -1)
        # x = x.squeeze(1)
        emb_time_1, emb_time_2 = time_emb
        # emb_time_1 = emb_time_1[0].unsqueeze(0).expand(x.shape[0], -1)
        # emb_time_2 = emb_time_2[0].unsqueeze(0).expand(x.shape[0], -1)
        x = self.linear_x(x.squeeze()) + self.linear_steps(steps.squeeze()) + self.linear_cond(cond.squeeze())# + self.linear_ent_emb(ent_embeddings.squeeze())# + self.linear_time_emb1(emb_time_1) + self.linear_time_emb2(emb_time_2)
        # x = torch.tanh(x)
        # x = F.silu(x)
        # x = F.leaky_relu(x, 0.4)
        # x = self.inp_drop(x)
        # x = self.linear(x)
        x = x.view(*shape)
        return x

# class EpsilonTheta(nn.Module): # 用来预测噪声的模型
#     def __init__(self, args, num_nodes, input_dropout=0, hidden_dropout=0, feat_dropout=0, channels=50, kernel_size=5, use_bias=True):
#         super().__init__()
#         self.num_nodes = num_nodes
#         self.i_enc = SinusoidalPositionEmbeddings(args.n_hidden)  # 扩散步数的encoder
#         # self.id_enc = PositionalEncoding(args.n_hidden, max_value=self.num_nodes)  # 使用定义的PositionalEncoding模块创建扩散步数编码器。

#         # self.inp_drop = torch.nn.Dropout(input_dropout)
#         # self.feature_map_drop = torch.nn.Dropout(feat_dropout)
#         self.conv1 = torch.nn.Conv1d(3, channels, kernel_size, stride=1,
#                                padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
#         # self.bn0 = torch.nn.BatchNorm1d(4)
#         # self.bn1 = torch.nn.BatchNorm1d(channels)
#         self.fc = nn.Linear(args.n_hidden*channels, args.n_hidden)#nn.Linear(args.n_hidden*channels, args.n_hidden)
#         nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))

#         # self.linear = nn.Linear(args.n_hidden*4, args.n_hidden)

#         # self.linear_x = nn.Linear(args.n_hidden, args.n_hidden)
#         # self.linear_steps = nn.Linear(args.n_hidden, args.n_hidden)
#         # self.linear_cond = nn.Linear(args.n_hidden, args.n_hidden)
#         # self.linear_ent_emb = nn.Linear(args.n_hidden, args.n_hidden)
#         # self.linear_time_emb1 = nn.Linear(args.n_hidden, args.n_hidden)
#         # self.linear_time_emb2 = nn.Linear(args.n_hidden, args.n_hidden)

#     def forward(self, x, steps, ent_embeddings, cond, time_emb):
#         shape = x.shape
#         steps = self.i_enc(steps.unsqueeze(1))
#         x = torch.cat([x, steps, cond], 1)
#         # x = torch.cat([x, steps, cond, ent_embeddings.unsqueeze(1)], 1)
#         # # x = torch.cat([x, steps, cond+ent_embeddings.unsqueeze(1)], 1)

#         # # # nodes_range = torch.arange(0, self.num_nodes, dtype=torch.int32, device=x.device).reshape(self.num_nodes, 1, 1)
#         # # # nodes_id_embs = self.id_enc(nodes_range)
#         # # steps = self.i_enc(steps.reshape(-1,1,1))
#         # # stacked_inputs = torch.cat([x, steps, cond], 1)
#         # # # stacked_inputs = self.bn0(stacked_inputs)
#         # # x = self.inp_drop(x)
#         x = self.conv1(x)
#         # # # x = self.bn1(x)
#         x = F.leaky_relu(x, 0.4)
#         # x = F.relu(x)
#         # # x = self.feature_map_drop(x)
#         x = x.view(shape[0], -1)


#         x = self.fc(x)
#         x = x.view(*shape)
#         return x

#         # shape = x.shape
#         # steps = self.i_enc(steps.unsqueeze(1))
#         # # x = torch.cat([x, steps, cond, ent_embeddings.unsqueeze(1)], -1)
#         # # x = x.squeeze(1)
#         # emb_time_1, emb_time_2 = time_emb
#         # # emb_time_1 = emb_time_1[0].unsqueeze(0).expand(x.shape[0], -1)
#         # # emb_time_2 = emb_time_2[0].unsqueeze(0).expand(x.shape[0], -1)
#         # x = self.linear_x(x.squeeze()) + self.linear_steps(steps.squeeze()) + self.linear_cond(cond.squeeze())# + self.linear_ent_emb(ent_embeddings.squeeze())# + self.linear_time_emb1(emb_time_1) + self.linear_time_emb2(emb_time_2)
#         # # x = self.linear(x)
#         # x = x.view(*shape)
#         # return x


    
class CondUpsampler(nn.Module):  # 初始化一个条件上采样器，通过两个线性层 (linear1 和 linear2) 将条件长度上采样到目标维度的一半，然后再上采样到目标维度。
    def __init__(self, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(target_dim, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)  # [3072,1,100]  ==> [3072,1,185]
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)  # [3072,1,370]
        x = F.leaky_relu(x, 0.4)
        return x
    
class EpsilonTheta_New(nn.Module): # 用来预测噪声的模型
    def __init__(self,
                 args,
                 residual_channels=8,
                 dilation_cycle_length=2,  # 空洞卷积的循环周期长度。
                 residual_layers=8,  # 残差块的层数。
                 ):
        super().__init__()
        self.args = args
        self.residual_hidden = args.n_hidden
        self.input_projection = nn.Conv1d(  # 输入序列的一维卷积层，用于将输入序列投影到残差块的通道数。
            1, residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.diffusion_embedding = SinusoidalPositionEmbeddings(args.n_hidden)  # 扩散步数的encoder
        self.cond_upsampler = CondUpsampler(self.args.n_hidden)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),  # dilation 参数是以 2 为底的指数，它的值按照 % dilation_cycle_length 循环变化，这是为了在多个残差块中使用不同的空洞卷积扩散率。
                    hidden_size=self.residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )

        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)  # 用于投影残差块中的跳跃连接。
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)  # 用于产生最终输出的卷积层。

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)
        

    def forward(self, inputs, time, ent_embeddings, cond):
        cond = ent_embeddings.unsqueeze(1)+cond
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x
    
class ResidualBlock(nn.Module):  # 学习输入序列中的非线性模式，并通过残差连接来传递有关输入的信息。跳跃连接 (skip) 也被保存，以供之后在网络中的其他位置使用。
    def __init__(self, hidden_size, residual_channels, dilation):  # 初始化一个残差块，包括一个空洞卷积层 (dilated_conv)、扩散函数投影 (diffusion_projection)、条件投影 (conditioner_projection) 以及输出层投影 (output_projection)。
        super().__init__()
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)  #  一个线性层，用于将输入 hidden_size 的特征投影到 residual_channels 的维度。这个层用于处理来自扩散函数的信息。
        self.conditioner_projection = nn.Conv1d(  # 一个卷积层，将输入的 1 个通道（conditioner）转换为 2 * residual_channels 个通道。这个层用于处理条件信息。
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.dilated_conv = nn.Conv1d(  #  一个卷积层，通过指定的 dilation 参数实现空洞卷积。这个卷积层将输入的 residual_channels 个通道转换为 2 * residual_channels 个通道，卷积核大小为 3。
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )

        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)  # 最终的输出投影层，将 residual_channels 个通道转换为 2 * residual_channels 个通道。

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):  # 接受输入 (x)、条件 (conditioner) 和扩散步数 (diffusion_step)，执行残差块的前向传播，返回残差块的输出和跳跃连接。
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)  # [3072,8,1]
        conditioner = self.conditioner_projection(conditioner)  # [3072,16,374]
        y = x + diffusion_step  # [3072,8,374]
        y = self.dilated_conv(y) + conditioner  # [3072,16,374]
        gate, filter = torch.chunk(y, 2, dim=1)  # [3072,16,374]  => [3072,8,374] and [3072,8,374]  y 被分割为两部分，gate 和 filter。这是为了实现门控和滤波操作
        y = torch.sigmoid(gate) * torch.tanh(filter)  # [3072,8,374]  通过 sigmoid 函数对 gate 进行门控，通过 tanh 函数对 filter 进行滤波。

        y = self.output_projection(y)  # [3072,16,374]
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)  # [3072,16,374]  => [3072,8,374] and [3072,8,374]   y 被再次分割为残差 (residual) 和跳跃连接 (skip)，最后返回 (x + residual) / math.sqrt(2.0), skip。
        return (x + residual) / math.sqrt(2.0), skip