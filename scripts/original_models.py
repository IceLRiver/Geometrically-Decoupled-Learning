import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PhysKANLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 32,
        learnable_omega: bool = True,
        omega_init: float = 1.0,
        use_residual_gate: bool = True,
        bias: bool = True,
    ):
        """
        in_dim:  输入维度 d
        out_dim: 输出维度 m
        hidden_dim: 残差 MLP 的隐藏层维度
        learnable_omega: sin/cos 的频率是否可学习
        omega_init: omega 初值
        use_residual_gate: 是否对残差项加 per-input gate (更稳)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual_gate = use_residual_gate

        # 物理基函数系数：每个输出维 j、每个输入维 i 各一组
        # shape: (out_dim, in_dim)
        self.w_x   = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.w_sin = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.w_cos = nn.Parameter(torch.zeros(out_dim, in_dim))

        # sin/cos 频率 omega：可以是标量，也可以是 per-input
        if learnable_omega:
            # per-input omega 更灵活，也更稳
            self.omega = nn.Parameter(torch.full((in_dim,), float(omega_init)))
        else:
            self.register_buffer("omega", torch.full((in_dim,), float(omega_init)))

        # 残差 MLP：共享结构，但对每个输入维都可用
        # 输入: (B, in_dim, 1) -> 输出: (B, in_dim, out_dim)
        self.res_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # 残差 gate：每个输入维一个 gate（广播到 out_dim）
        # 让残差从小开始，训练更稳定
        if use_residual_gate:
            self.res_gate = nn.Parameter(torch.zeros(in_dim))  # 初值 0 => gate ~ 0
        else:
            self.res_gate = None

        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        # 物理项系数用小值初始化（避免一上来过强振荡）
        nn.init.normal_(self.w_x,   mean=0.0, std=0.02)
        nn.init.normal_(self.w_sin, mean=0.0, std=0.02)
        nn.init.normal_(self.w_cos, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # 残差 MLP 默认初始化即可；gate 初始为 0 (若启用)
        if self.res_gate is not None:
            nn.init.zeros_(self.res_gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim)
        return: (B, out_dim)
        """
        # 物理基函数
        # (B, in_dim)
        x_lin = x
        x_sin = torch.sin(x * self.omega)   # omega 会广播 (in_dim,)
        x_cos = torch.cos(x * self.omega)

        # 将 (B, in_dim) 与 (out_dim, in_dim) 做按 input 维的加权求和：
        # 计算：sum_i w[:,i]*f(x_i)
        # 用 einsum 很直接：'bi,oi->bo'
        y_phys = (
            torch.einsum("bi,oi->bo", x_lin, self.w_x) +
            torch.einsum("bi,oi->bo", x_sin, self.w_sin) +
            torch.einsum("bi,oi->bo", x_cos, self.w_cos)
        )

        # 残差项：对每个输入维 x_i 做 MLP_res(x_i) 得到 (B, in_dim, out_dim)，然后 sum_i
        x_exp = x.unsqueeze(-1)  # (B, in_dim, 1)
        y_res_all = self.res_mlp(x_exp)  # (B, in_dim, out_dim)

        if self.use_residual_gate:
            # gate: (in_dim,) -> (1, in_dim, 1) -> broadcast
            gate = torch.tanh(self.res_gate).view(1, self.in_dim, 1)
            y_res_all = y_res_all * gate

        y_res = y_res_all.sum(dim=1)  # (B, out_dim)

        y = y_phys + y_res
        if self.bias is not None:
            y = y + self.bias
        return y


class OrbitPhysKAN(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        seq_len: int = 38,
        d_model: int = 128,
        kan_hidden: int = 32,
        learnable_omega: bool = True,
        omega_init: float = 1.0,
        use_residual_gate: bool = True,
        dropout: float = 0.1,
        temporal_pool: str = "attn",
        expand_factor: int = 7,
        num_mid_layers: int = 4,
        norm_params: dict = None,   # 标准化参数字典
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.temporal_pool = temporal_pool
        self.expand_factor = expand_factor
        self.num_mid_layers = num_mid_layers

        # 注册标准化参数（全部转为 float32 buffer）
        if norm_params is not None:
            self.register_buffer('cam_mean', norm_params['cam_mean'].to(torch.float32).view(1, 3))
            self.register_buffer('cam_std',  norm_params['cam_std'].to(torch.float32).view(1, 3))
            self.register_buffer('los_mean', norm_params['los_mean'].to(torch.float32).view(1, 3))
            self.register_buffer('los_std',  norm_params['los_std'].to(torch.float32).view(1, 3))
            self.register_buffer('rho_mean', torch.tensor(norm_params['rho_mean'], dtype=torch.float32).view(1, 1))
            self.register_buffer('rho_std',  torch.tensor(norm_params['rho_std'], dtype=torch.float32).view(1, 1))
            self.register_buffer('pos_mean', norm_params['pos_mean'].to(torch.float32).view(1, 3))
            self.register_buffer('pos_std',  norm_params['pos_std'].to(torch.float32).view(1, 3))
            self.register_buffer('vel_mean', norm_params['vel_mean'].to(torch.float32).view(1, 3))
            self.register_buffer('vel_std',  norm_params['vel_std'].to(torch.float32).view(1, 3))
        else:
            for name in ['cam_mean','cam_std','los_mean','los_std','rho_mean','rho_std','pos_mean','pos_std','vel_mean','vel_std']:
                self.register_buffer(name, None)

        # 网络结构
        self.expanded_dim = input_dim * expand_factor
        self.combined_dim = self.expanded_dim * 3
        self.embed = PhysKANLayer(self.combined_dim, d_model, kan_hidden, learnable_omega, omega_init, use_residual_gate, True)
        if num_mid_layers > 0:
            self.mid_layers = nn.ModuleList([
                PhysKANLayer(d_model, d_model, kan_hidden, learnable_omega, omega_init, use_residual_gate, True)
                for _ in range(num_mid_layers)
            ])
        else:
            self.mid_layers = None
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if temporal_pool == "attn":
            self.attn_score = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        else:
            self.attn_score = None
        self.f1 = PhysKANLayer(d_model, 1, kan_hidden, learnable_omega, omega_init, use_residual_gate, True)
        self.f2 = PhysKANLayer(d_model + 3, 3, kan_hidden, learnable_omega, omega_init, use_residual_gate, True)

    def expand_features(self, x):
        eps = 1e-8
        a = x
        a2, a3 = a ** 2, a ** 3
        sin_term = torch.sin(2 * math.pi * a)
        cos_term = torch.cos(2 * math.pi * a)
        ln_term = torch.log(torch.abs(a) + eps)
        exp_term = torch.exp(a)
        return torch.cat([a, a2, a3, sin_term, cos_term, ln_term, exp_term], dim=-1)

    def forward(self, x):
        B = x.size(0)
        x_3d = x.view(B, 40, self.input_dim)
        x_exp = self.expand_features(x_3d)
        comb_list = []
        for n in range(2, 40):
            triple = x_exp[:, [0, n-1, 39], :]
            comb_list.append(triple.reshape(B, -1))
        combs = torch.stack(comb_list, dim=1)

        h = torch.stack([self.embed(combs[:, t]) for t in range(self.seq_len)], dim=1)
        if self.mid_layers is not None:
            B_, T_, D_ = h.shape
            h_flat = h.view(B_ * T_, D_)
            for layer in self.mid_layers:
                h_flat = h_flat + layer(h_flat)
            h = h + h_flat.view(B_, T_, D_)
        h = self.drop(self.norm(h))

        if self.temporal_pool == "mean":
            h_agg = h.mean(dim=1)
        elif self.temporal_pool == "max":
            h_agg = h.max(dim=1).values
        else:  # attn
            alpha = torch.softmax(self.attn_score(h).squeeze(-1), dim=1).unsqueeze(-1)
            h_agg = (h * alpha).sum(dim=1)

        rho_norm = self.f1(h_agg)
        cam_norm = x_3d[:, 39, :3]
        los_norm = x_3d[:, 39, 3:6]

        if self.cam_mean is not None:
            cam_phys = cam_norm * self.cam_std + self.cam_mean
            los_phys = los_norm * self.los_std + self.los_mean
            los_phys = los_phys / (torch.norm(los_phys, dim=1, keepdim=True) + 1e-8)
            rho_phys = rho_norm * self.rho_std + self.rho_mean
        else:
            cam_phys, los_phys, rho_phys = cam_norm, los_norm, rho_norm

        xyz_phys = cam_phys + rho_phys * los_phys
        vxyz_phys = self.f2(torch.cat([h_agg, xyz_phys], dim=1))

        return {'rho': rho_phys, 'xyz': xyz_phys, 'vxyz': vxyz_phys}

