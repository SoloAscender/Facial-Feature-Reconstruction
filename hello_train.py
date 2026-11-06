import torch
import torch.nn as nn

model = nn.Linear(1,1)

print("Before Training:")
print("Weight:", model.weight.item(), "Bias:", model.bias.item())

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[4.0], [7.0], [10.0], [13.0]])

optimiser = torch.optim.SGD(model.parameters(), lr = 0.01)

for step in range(200):
    prediction = model(x)
    loss = ((prediction - y)**2).mean()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if step%50 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

print("After Training:")
print("Weight:", model.weight.item(), "Bias:", model.bias.item())


torch.save(model.state_dict(), "checkpoint.pt")
print("Saved checkpoint.pt")

new_model = nn.Linear(1,1)
new_model.load_state_dict(torch.load("checkpoint.pt"))
print("Reloaded weight:", new_model.weight.item())