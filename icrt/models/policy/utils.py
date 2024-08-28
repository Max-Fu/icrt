import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResNetBlock(nn.Module):
    def __init__(self, dim, activation, dropout_prob, use_layer_norm):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob else nn.Identity()
        self.norm = nn.LayerNorm(self.dim) if use_layer_norm else nn.Identity()
        self.mlp = Mlp(self.dim, 4 * self.dim, act_layer=activation)

    def forward(self, x):
        residual = x
        x = self.norm(self.dropout(x))
        x = self.mlp(x)
        return residual + x

class MLPResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks, activation, dropout_prob, use_layer_norm):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        residual_blocks = [
            ResNetBlock(hidden_dim, activation, dropout_prob, use_layer_norm) for _ in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*residual_blocks)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.activation = activation()

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        return self.decoder(self.activation(x))

class ConditionedDiffusion(nn.Module):
    def __init__(
        self,
        output_dim,
        hidden_dim,
        cond_dim,
        train_noise_scheduler,
        inference_noise_scheduler,
        activation=nn.SiLU,
        time_dim=32,
        num_blocks=4,
        dropout_prob=0.1,
        use_layer_norm=True,
        inference_steps=None
    ):
        super().__init__()

        self.output_dim = output_dim

        self.train_noise_scheduler = train_noise_scheduler
        self.inference_noise_scheduler = inference_noise_scheduler
        
        self.time_encoder = nn.Sequential(SinusoidalPosEmb(time_dim), Mlp(time_dim, 4 * time_dim))
        self.model = MLPResNet(time_dim + cond_dim + output_dim, 
                               hidden_dim, output_dim, 
                               num_blocks, activation, dropout_prob, use_layer_norm)
        if inference_steps is None:
            inference_steps = self.inference_noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = inference_steps

    def forward(self, sample, t, condition):
        B = sample.shape[0]
        time_features = self.time_encoder(t)
        x = torch.cat([time_features, condition, sample], dim=-1)
        return self.model(x)

    def calculate_loss(self, latents: torch.Tensor, label: torch.Tensor):
        # get dimensions 
        B = latents.shape[0]
        noise = torch.randn(label.shape, device=label.device)
        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps, 
            (B,), device=label.device
        ).long()
        
        noisy_label = self.train_noise_scheduler.add_noise(
            label, noise, timesteps)

        pred = self(noisy_label, timesteps, latents)

        # calculate the loss
        target = noise
        loss = F.mse_loss(pred, target)
        return loss


    def conditional_sample(self, latents: torch.Tensor):
        bsz, latent_seqlen = latents.shape[:2]

        prediction = torch.randn((bsz, self.output_dim), dtype=latents.dtype, device=latents.device)

        self.inference_noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.inference_noise_scheduler.timesteps:
            model_output = self(prediction, t.expand(bsz).to(latents.device), latents)
            prediction = self.inference_noise_scheduler.step(model_output, t, prediction).prev_sample
        
        return prediction