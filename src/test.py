import torch
from src.models.mlp_model import MLP
from src.utils.utils import compute_accuracy
from src.data_processing.load_data import get_data_loaders
import os
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model
test_dir = '../data/raw/test'
classes = os.listdir(test_dir)

img_height, img_width = 128, 128
input_dims = img_height * img_width
output_dims = len(classes)
hidden_dims = 64

model = MLP(input_dims, hidden_dims, output_dims).to(device)
model.load_state_dict(torch.load('./saved_models/best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Prepare test data
label2idx = {cls: idx for idx, cls in enumerate(classes)}
test_loader = get_data_loaders(train_dir='../data/raw/train', test_dir=test_dir, label2idx=label2idx, batch_size=256)[2]

test_target = []
test_predict = []
model.eval()
with torch.no_grad():
  for X_samples, y_samples in test_loader:
    X_samples = X_samples.to(device)
    y_samples = y_samples.to(device)
    outputs = model(X_samples)

    test_predict.append(outputs.cpu())
    test_target.append(y_samples.cpu())

  test_predict = torch.cat(test_predict)
  test_target = torch.cat(test_target)
  test_acc = compute_accuracy(test_predict, test_target)

  print('Evaluation on test set:')
  print(f'Accuracy: {test_acc}')