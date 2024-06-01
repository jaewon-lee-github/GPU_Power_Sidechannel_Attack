import torch
import torch.nn.functional as F

y_true = torch.tensor([0, 1, 2])  # Class indices
y_pred = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float32)  # Predicted probabilities

loss = F.cross_entropy(y_pred, y_true)
print(loss.item())