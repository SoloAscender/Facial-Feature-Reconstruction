import torch
import torch.nn as nn
import math

from fal import FrequencyAwareLoss

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
    def __init__(self, channels=4, t_dim=16):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.channel_mix = nn.Linear(channels, channels)
        self.t_proj = nn.Linear(t_dim, channels)

    def forward(self, noisy_latent, t_embed):
        x = self.conv(noisy_latent)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.channel_mix(x)
        x = x.permute(0, 4, 1, 2, 3)
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
    p.requires_grad = False

backbone.channel_mix = LoRALinear(backbone.channel_mix, rank=4, alpha=8)

rrah = RRAH()
fal = FrequencyAwareLoss(q=50.0, gamma=2.0)

trainable_params = [p for p in backbone.channel_mix.parameters() if p.requires_grad]
trainable_params += list(rrah.parameters())

optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

conv_weight_before = backbone.conv.weight.clone()
base_channel_mix_weight_before = backbone.channel_mix.base.weight.clone()

# single toy (LQ, HQ) pair, shared across all sampled (t, noise) draws this run
lq_latent_single = torch.randn(1, 4, 5, 16, 16)
target_latent_single = torch.randn(1, 4, 5, 16, 16)  # x_1

FAL_WEIGHT = 1.0
BATCH = 64         # number of random (t, x0) draws per epoch, averaged for a stable loss curve

for epoch in range(50):
    # fresh random draws each epoch: K independent (t, noise) samples for the same (LQ,HQ) pair
    t = torch.rand(BATCH)                                   # t ~ U(0,1), one per sample
    x0 = torch.randn(BATCH, 4, 5, 16, 16)                    # fresh noise per sample

    lq_latent = lq_latent_single.expand(BATCH, -1, -1, -1, -1)
    target_latent = target_latent_single.expand(BATCH, -1, -1, -1, -1)

    t_broadcast = t.view(BATCH, 1, 1, 1, 1)
    x_t = (1 - t_broadcast) * x0 + t_broadcast * target_latent   # direct interpolation, no rollout

    t_embed = timestep_embedding(t)

    velocity_pred = backbone(x_t, t_embed)
    velocity_pred = rrah(lq_latent, velocity_pred, t_embed)

    v_target = target_latent - x0
    fm_loss = ((velocity_pred - v_target) ** 2).mean()

    z_hat_t = x_t + (1 - t_broadcast) * velocity_pred
    freq_loss = fal(z_hat_t, target_latent)

    loss = fm_loss + FAL_WEIGHT * freq_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0 or epoch == 49:
        print(f"epoch {epoch}: total={loss.item():.4f}  fm={fm_loss.item():.4f}  freq={freq_loss.item():.4f}")

print("\nDid frozen conv weight change?", not torch.equal(conv_weight_before, backbone.conv.weight))
print("Did frozen base channel_mix weight change?", not torch.equal(base_channel_mix_weight_before, backbone.channel_mix.base.weight))

# ---- separate eval: multi-step sampling (t=0 -> t=1) using the trained model ----
with torch.no_grad():
    num_sample_steps = 5
    dt = 1.0 / num_sample_steps
    x_eval = torch.randn(1, 4, 5, 16, 16)
    for step in range(num_sample_steps):
        t_val = step / num_sample_steps
        t_eval = torch.tensor([t_val])
        t_embed_eval = timestep_embedding(t_eval)
        v = backbone(x_eval, t_embed_eval)
        v = rrah(lq_latent_single, v, t_embed_eval)
        x_eval = x_eval + v * dt
    print("Eval-time sampled output vs target_latent MSE:",
          ((x_eval - target_latent_single) ** 2).mean().item())