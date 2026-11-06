import torch
import torch.nn as nn

Q_BASE = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=torch.float32)


def dct_matrix(n=8):
    k = torch.arange(n).float().unsqueeze(1)
    i = torch.arange(n).float().unsqueeze(0)
    basis = torch.cos(torch.pi / n * (i + 0.5) * k)
    basis[0] *= 1.0 / torch.sqrt(torch.tensor(2.0))
    basis *= torch.sqrt(torch.tensor(2.0 / n))
    return basis


def block_dct_8x8(x, B):
    """x: [..., H, W], H and W divisible by 8. B: [8,8] DCT basis matrix."""
    *lead, H, W = x.shape
    assert H % 8 == 0 and W % 8 == 0, "H and W must be divisible by 8"
    x = x.reshape(*lead, H // 8, 8, W // 8, 8)
    x = x.permute(*range(len(lead)), -4, -2, -3, -1)
    x = torch.einsum('ij,...jk->...ik', B, x)
    x = torch.einsum('...ij,kj->...ik', x, B)
    x = x.permute(*range(len(lead)), -4, -2, -3, -1)
    x = x.reshape(*lead, H, W)
    return x


class FrequencyAwareLoss(nn.Module):
    """
    FAL from FADRA (eqs 3-6). Operates on the reconstructed latent z_t (not velocity).
    q: JPEG quality factor in [1,100], controls how strongly HVS weighting is applied
       (q->99 flattens toward uniform weighting; q=50 uses the base table as-is).
    gamma: sharpens the weight contrast between high- and low-sensitivity frequencies.
    Both kept as plain floats (not learned) so they stay easy to sweep later.
    """
    def __init__(self, q=50.0, gamma=2.0):
        super().__init__()
        self.q = q
        self.gamma = gamma
        self.register_buffer('dct_basis', dct_matrix(8))
        self.register_buffer('weight', self._build_weight(q, gamma))

    def _build_weight(self, q, gamma):
        Q_cur = torch.clamp(torch.floor((Q_BASE * (100 - q) + 25) / 50), min=1)
        Qbar = Q_cur.mean()
        w_tilde = (Qbar / Q_cur) ** gamma
        w = w_tilde / w_tilde.mean()
        return w  # [8,8]

    def set_params(self, q=None, gamma=None):
        """Allows sweeping q/gamma without rebuilding the module."""
        self.q = q if q is not None else self.q
        self.gamma = gamma if gamma is not None else self.gamma
        self.weight = self._build_weight(self.q, self.gamma).to(self.weight.device)

    def forward(self, z_hat, z):
        """
        z_hat, z: [B, C, T, H, W] latents (H, W divisible by 8).
        Returns scalar loss: mean_{B,C,T,H,W} [ w(H,W) * (Z_hat - Z)^2 ]
        """
        assert z_hat.shape == z.shape, "prediction and target latents must match shape"
        B_, C_, T_, H_, W_ = z_hat.shape

        Z_hat = block_dct_8x8(z_hat, self.dct_basis)
        Z = block_dct_8x8(z, self.dct_basis)

        # tile the 8x8 weight table across the full H x W latent, then broadcast over B,C,T
        n_blocks_h = H_ // 8
        n_blocks_w = W_ // 8
        w_tiled = self.weight.repeat(n_blocks_h, n_blocks_w)          # [H, W]
        w_tiled = w_tiled.view(1, 1, 1, H_, W_)                        # broadcastable

        loss = w_tiled * (Z_hat - Z) ** 2
        return loss.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    fal = FrequencyAwareLoss(q=50.0, gamma=2.0)

    z = torch.randn(1, 4, 5, 16, 16)

    # sanity 1: identical inputs -> zero loss
    loss_same = fal(z, z)
    print("Loss when z_hat == z (should be ~0):", loss_same.item())

    # sanity 2: different inputs -> positive loss
    z_hat = torch.randn(1, 4, 5, 16, 16)
    loss_diff = fal(z_hat, z)
    print("Loss for random z_hat != z (should be > 0):", loss_diff.item())

    # sanity 3: gradient flows back to z_hat
    z_hat_grad = torch.randn(1, 4, 5, 16, 16, requires_grad=True)
    loss_grad = fal(z_hat_grad, z)
    loss_grad.backward()
    print("Gradient exists on z_hat:", z_hat_grad.grad is not None)
    print("Gradient norm:", z_hat_grad.grad.norm().item())

    # sanity 4: as z_hat moves closer to z, loss should decrease monotonically
    z_hat_far = z + torch.randn_like(z) * 5.0
    for alpha in [1.0, 0.5, 0.1, 0.0]:
        z_hat_interp = z + alpha * (z_hat_far - z)
        l = fal(z_hat_interp, z).item()
        print(f"alpha={alpha}: loss={l:.4f}")

    # sanity 5: set_params sweep works without rebuilding
    fal.set_params(q=90, gamma=1)
    print("After set_params(q=90, gamma=1), weight range:",
          fal.weight.min().item(), fal.weight.max().item())