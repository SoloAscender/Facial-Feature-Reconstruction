import torch
import torch.nn as nn

torch.manual_seed(42)

class LoRALinear(nn.Module):
    """
    Wraps an existing frozen nn.Linear layer with a small trainable
    low-rank correction, instead of fine-tuning the whole thing.
    """
    def __init__(self, base_linear: nn.Linear, rank=4, alpha=8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False  # freeze the original weight matrix entirely

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)  # small random init
        self.B = nn.Parameter(torch.zeros(out_features, rank))        # all-zero init -- deliberate
        self.scale = alpha / rank

    def forward(self, x):
        base_out = self.base(x)                              # frozen path, untouched
        lora_out = (x @ self.A.T) @ self.B.T * self.scale     # small trainable correction path
        return base_out + lora_out


base_layer = nn.Linear(16, 16)
lora_layer = LoRALinear(base_layer, rank=4, alpha=8)

x = torch.randn(2, 16)

with torch.no_grad():
    base_only = base_layer(x)
    combined = lora_layer(x)

print("Difference before training (base vs LoRA-wrapped):", (combined - base_only).abs().max().item())

for name, p in lora_layer.named_parameters():
    print(name, "requires_grad =", p.requires_grad, "shape:", tuple(p.shape))


# ---- Prove training actually updates A and B, and leaves base frozen ----
target = torch.randn(2, 16)  # some arbitrary thing to train toward
optimizer = torch.optim.SGD([lora_layer.A, lora_layer.B], lr=0.1)

base_weight_before = lora_layer.base.weight.clone()

for step in range(50):
    output = lora_layer(x)
    loss = ((output - target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("\nAfter training:")
print("Loss:", loss.item())
print("Did base.weight change at all?", not torch.equal(base_weight_before, lora_layer.base.weight))
print("B is still all zeros?", torch.equal(lora_layer.B, torch.zeros_like(lora_layer.B)))