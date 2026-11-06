import torch
import torch.nn as nn

model = nn.Linear(1,1)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

x = torch.tensor([[5.0]])
prediction = model(x)
print("Prediction for x = 5:", prediction.item())