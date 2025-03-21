import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 2)
        self.sigmoid = nn.Sigmoid()

        with torch.no_grad():
            self.hidden.weight = nn.Parameter(torch.tensor([[0.15, 0.20], [0.25, 0.30]]))
            self.hidden.bias = nn.Parameter(torch.tensor([0.35, 0.35]))
            self.output.weight = nn.Parameter(torch.tensor([[0.40, 0.45], [0.50, 0.55]]))
            self.output.bias = nn.Parameter(torch.tensor([0.60, 0.60]))

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = SimpleNN()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

X = torch.tensor([[0.05, 0.10]])
y = torch.tensor([[0.01, 0.99]])

output = model(X)
loss = criterion(output, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Updated weights (input to hidden):")
print(model.hidden.weight.data)
print("Updated weights (hidden to output):")
print(model.output.weight.data)
