import torch

def compute_accuracy(y_hat, y_true):
  _, y_hat = torch.max(y_hat, 1)
  correct = (y_hat == y_true).sum().item()
  accuracy = correct / len(y_true)
  return accuracy