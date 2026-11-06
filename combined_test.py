import torch
import torch.nn as nn
import math

torch.manual_seed(42)

def timestep_embedding(t, dim=16):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank=4, alpha=8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.A.T) @ self.B.T * self.scale
        return base_out + lora_out


class FrozenBackboneStub(nn.Module):
    """
    conv = spatial mixing, stays plainly frozen, no adapter at all.
    channel_mix = stand-in for an attention/projection layer --
    THIS is what gets wrapped with LoRA, mirroring real Wan2.1 fine-tuning.
    """
    def __init__(self, channels=4, t_dim=16):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.channel_mix = nn.Linear(channels, channels)
        self.t_proj = nn.Linear(t_dim, channels)

    def forward(self, noisy_latent, t_embed):
        x = self.conv(noisy_latent)
        x = x.permute(0, 2, 3, 4, 1)          # channels moved to last dim -- nn.Linear needs that
        x = self.channel_mix(x)
        x = x.permute(0, 4, 1, 2, 3)          # move channels back
        shift = self.t_proj(t_embed)[:, :, None, None, None]
        return x + shift


class RRAH(nn.Module):
    def __init__(self, channels=4, t_dim=16):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        self.t_proj = nn.Linear(t_dim, channels)

    def forward(self, lq_latent, velocity_pred, t_embed):
        combined = torch.cat([lq_latent, velocity_pred], dim=1)
        residual = self.refine(combined)
        shift = self.t_proj(t_embed)[:, :, None, None, None]
        residual = residual + shift
        return velocity_pred + residual


# ---- Assemble ----
backbone = FrozenBackboneStub()
for p in backbone.parameters():
    p.requires_grad = False          # freeze the whole backbone first

backbone.channel_mix = LoRALinear(backbone.channel_mix, rank=4, alpha=8)  # then wrap just this layer

rrah = RRAH()

trainable_params = [p for p in backbone.channel_mix.parameters() if p.requires_grad]
trainable_params += list(rrah.parameters())

frozen_count = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
trainable_count = sum(p.numel() for p in trainable_params)
print(f"Frozen params: {frozen_count}")
print(f"Trainable params (LoRA + RRAH): {trainable_count}\n")

optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

conv_weight_before = backbone.conv.weight.clone()
base_channel_mix_weight_before = backbone.channel_mix.base.weight.clone()

lq_latent = torch.randn(1, 4, 5, 16, 16)
target_latent = torch.randn(1, 4, 5, 16, 16)

fixed_start_noise = torch.randn(1,4,5,16,16)

num_steps = 5
dt = 1.0 / num_steps

for epoch in range(50):
    noisy_latent = fixed_start_noise.clone()
    for step in range(num_steps):
        t = torch.tensor([step / num_steps])
        t_embed = timestep_embedding(t)
        velocity_pred = backbone(noisy_latent, t_embed)
        velocity_pred = rrah(lq_latent, velocity_pred, t_embed)
        noisy_latent = noisy_latent - velocity_pred * dt

    loss = ((noisy_latent - target_latent) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%5 == 0 or epoch == 49:
        print(f"epoch {epoch}: loss {loss.item():.4f}")

print("\nDid frozen conv weight change?", not torch.equal(conv_weight_before, backbone.conv.weight))
print("Did frozen base channel_mix weight change?", not torch.equal(base_channel_mix_weight_before, backbone.channel_mix.base.weight))