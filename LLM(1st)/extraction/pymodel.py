import torch
import torch.nn as nn
import torch.optim as optim

class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(1,1)
    def forward(self,x):
        return self.Linear(x)
    
x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[2.],[4.],[6.],[8.]])

model = mymodel()

loss_fn = nn.MSELoss()

optimiser = optim.Adam(model.parameters(),lr=0.1)

for epoch in range(10):
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()
    print(f"EPOCH : {epoch + 1}, LOSS : {loss.item():.4f}")

#if wanna save then uncomment this line ....
"""
torch.save(model.state-dict(), "Linear-model_using_pytorch.pth")
"""
    