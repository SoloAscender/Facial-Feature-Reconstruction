import torch
import torch.nn as nn
import math

torch.manual_seed(42)

def timestep_embedding(t, dim = 16):
    half = dim//2
    freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim =-1)


#Assuming we have a Wan2.1 Model here, currently we will use a tiny substitute for the test

class FrozenBackboneStub(nn.Module):
    def __init__(self, channels = 4, t_dim = 16):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(t_dim, channels)

    def forward(self, noisy_latent, t_embed):
        shift = self.t_proj(t_embed)
        shift = shift[:, :, None, None, None]
        return self.conv(noisy_latent) + shift


class RRAH(nn.Module):
    def __init__(self, channels = 4, t_dim = 16):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        self.t_proj = nn.Linear(t_dim, channels)
        

    def forward(self, lq_latent, velocity_pred, t_embed):
        combined = torch.cat([lq_latent, velocity_pred], dim = 1)
        residual = self.refine(combined)
        shift = self.t_proj(t_embed)[:, :, None, None, None]
        residual = residual + shift
        return velocity_pred + residual


backbone = FrozenBackboneStub()
for param in backbone.parameters():
    param.requires_grad = False

rrah = RRAH()


frozen_params = sum(p.numel() for p in backbone.parameters())
trainable_params = sum(p.numel() for p in rrah.parameters())
print(f"Frozen backbone params: {frozen_params}")
print(f"Trainable RRAH params: {trainable_params}\n")

lq_latent = torch.randn(1,4,5,16,16)
noisy_latent = torch.randn(1,4,5,16,16)

num_steps = 5
dt = 1.0 / num_steps

print("Simulating a flow-matching sampling loop with RRAH correction at every step:\n")

for step in range(num_steps):
    t = torch.tensor([step/num_steps])
    t_embed = timestep_embedding(t)

    velocity_pred = backbone(noisy_latent, t_embed)
    velocity_pred = rrah(lq_latent, velocity_pred, t_embed)

    noisy_latent = noisy_latent - velocity_pred * dt

    print(f"step {step}: velocity_pred shape {tuple(velocity_pred.shape)}, "
          f"noisy_latent mean {noisy_latent.mean().item():.4f}")

print("\n Final restored latent shape:", noisy_latent.shape)



