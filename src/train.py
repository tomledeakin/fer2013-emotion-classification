# # Import necessary modules
# from src.models.mlp_model import MLP
# import os
# import torch
# import torch.nn as nn
# from src.data_processing.load_data import get_data_loaders
# from src.utils.utils import compute_accuracy
# from tqdm import tqdm  # Import tqdm
#
# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
#
# # Data directories
# train_dir = '../data/raw/train'
# test_dir = '../data/raw/test'
#
# # Get class names
# classes = os.listdir(test_dir)
# print("Classes:", classes)
#
# # Label encoding
# label2idx = {cls: idx for idx, cls in enumerate(classes)}
# idx2label = {idx: cls for idx, cls in enumerate(classes)}
#
# # Model parameters
# img_height, img_width = 128, 128
# input_dims = img_height * img_width
# output_dims = len(classes)
# hidden_dims = 64
# lr = 1e-3
#
# # Initialize model, loss, optimizer
# model = MLP(input_dims, hidden_dims, output_dims).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
# # Get data loaders
# train_loader, val_loader, test_loader = get_data_loaders(
#     train_dir=train_dir,
#     test_dir=test_dir,
#     label2idx=label2idx,
#     batch_size=256
# )
#
# # Training setup
# epochs = 100
# train_losses = []
# val_losses = []
# train_accs = []
# val_accs = []
# best_val_acc = 0.0
#
# # Training loop
# for epoch in range(epochs):
#     # Training Phase
#     model.train()  # Set model to training mode
#     train_loss = 0.0
#     train_target = []
#     train_predict = []
#
#     # Wrap train_loader with tqdm for progress bar
#     train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Training]', leave=True)
#
#     for batch_idx, (X_samples, y_samples) in enumerate(train_loader_tqdm, 1):
#         X_samples = X_samples.to(device)
#         y_samples = y_samples.to(device)
#         optimizer.zero_grad()
#         outputs = model(X_samples)
#         loss = criterion(outputs, y_samples)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#
#         train_target.append(y_samples.cpu())
#         train_predict.append(outputs.cpu())
#
#         # Update tqdm description with current loss
#         train_loader_tqdm.set_postfix({'Batch Loss': loss.item()})
#
#     # Calculate average training loss and accuracy
#     train_loss /= len(train_loader)
#     train_losses.append(train_loss)
#     train_target = torch.cat(train_target)
#     train_predict = torch.cat(train_predict)
#     train_acc = compute_accuracy(train_predict, train_target)
#     train_accs.append(train_acc)
#
#     # Validation Phase
#     model.eval()  # Set model to evaluation mode
#     val_loss = 0.0
#     val_target = []
#     val_predict = []
#
#     # Wrap val_loader with tqdm for progress bar
#     val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Validation]', leave=True)
#
#     with torch.no_grad():
#         for X_samples, y_samples in val_loader_tqdm:
#             X_samples = X_samples.to(device)
#             y_samples = y_samples.to(device)
#             outputs = model(X_samples)
#             loss = criterion(outputs, y_samples)
#             val_loss += loss.item()
#             val_target.append(y_samples.cpu())
#             val_predict.append(outputs.cpu())
#
#             # Update tqdm description with current loss
#             val_loader_tqdm.set_postfix({'Batch Loss': loss.item()})
#
#     # Calculate average validation loss and accuracy
#     val_loss /= len(val_loader)
#     val_losses.append(val_loss)
#     val_target = torch.cat(val_target)
#     val_predict = torch.cat(val_predict)
#     val_acc = compute_accuracy(val_predict, val_target)
#     val_accs.append(val_acc)
#
#     # Print epoch statistics
#     print(f'\nEPOCH {epoch + 1}/{epochs}: '
#           f'Training Loss: {train_loss:.3f} | Validation Loss: {val_loss:.3f}')
#     print(f'Training Accuracy: {train_acc:.3f} | Validation Accuracy: {val_acc:.3f}')
#
#     # Save the best model if validation accuracy improves
#     save_dir = 'outputs'
#     os.makedirs(save_dir, exist_ok=True)
#
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         model_save_path = os.path.join(save_dir, 'best_model.pth')
#         print(f"Validation accuracy improved. Saving the best model at {model_save_path}")
#         torch.save(model.state_dict(), model_save_path)
#         print('------------------------------------------------------------------------')


# Import necessary modules
from src.models.cnn_model import EmotionCNN  # Use CNN model
import os
import torch
import torch.nn as nn
from src.data_processing.load_data import get_data_loaders
from src.utils.utils import compute_accuracy
from tqdm import tqdm  # Import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Data directories
train_dir = '../data/raw/train'
test_dir = '../data/raw/test'

# Get class names
classes = os.listdir(test_dir)
print("Classes:", classes)

# Label encoding
label2idx = {cls: idx for idx, cls in enumerate(classes)}
idx2label = {idx: cls for idx, cls in enumerate(classes)}

# Model parameters
output_dims = len(classes)  # Number of output classes
lr = 1e-3

# Initialize CNN model
model = EmotionCNN(num_classes=output_dims).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Get data loaders
train_loader, val_loader, test_loader = get_data_loaders(
    train_dir=train_dir,
    test_dir=test_dir,
    label2idx=label2idx,
    batch_size=256
)

# Training setup
epochs = 100
train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_acc = 0.0

# Training loop
for epoch in range(epochs):
    # Training Phase
    model.train()  # Set model to training mode
    train_loss = 0.0
    train_target = []
    train_predict = []

    # Wrap train_loader with tqdm for progress bar
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Training]', leave=True)

    for batch_idx, (X_samples, y_samples) in enumerate(train_loader_tqdm, 1):
        X_samples = X_samples.to(device)
        y_samples = y_samples.to(device)
        optimizer.zero_grad()
        outputs = model(X_samples)
        loss = criterion(outputs, y_samples)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_target.append(y_samples.cpu())
        train_predict.append(outputs.cpu())

        # Update tqdm description with current loss
        train_loader_tqdm.set_postfix({'Batch Loss': loss.item()})

    # Calculate average training loss and accuracy
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    train_target = torch.cat(train_target)
    train_predict = torch.cat(train_predict)
    train_acc = compute_accuracy(train_predict, train_target)
    train_accs.append(train_acc)

    # Validation Phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_target = []
    val_predict = []

    # Wrap val_loader with tqdm for progress bar
    val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Validation]', leave=True)

    with torch.no_grad():
        for X_samples, y_samples in val_loader_tqdm:
            X_samples = X_samples.to(device)
            y_samples = y_samples.to(device)
            outputs = model(X_samples)
            loss = criterion(outputs, y_samples)
            val_loss += loss.item()
            val_target.append(y_samples.cpu())
            val_predict.append(outputs.cpu())

            # Update tqdm description with current loss
            val_loader_tqdm.set_postfix({'Batch Loss': loss.item()})

    # Calculate average validation loss and accuracy
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_target = torch.cat(val_target)
    val_predict = torch.cat(val_predict)
    val_acc = compute_accuracy(val_predict, val_target)
    val_accs.append(val_acc)

    # Print epoch statistics
    print(f'\nEPOCH {epoch + 1}/{epochs}: '
          f'Training Loss: {train_loss:.3f} | Validation Loss: {val_loss:.3f}')
    print(f'Training Accuracy: {train_acc:.3f} | Validation Accuracy: {val_acc:.3f}')

    # Save the best model if validation accuracy improves
    save_dir = 'outputs'
    os.makedirs(save_dir, exist_ok=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_save_path = os.path.join(save_dir, 'best_model.pth')
        print(f"Validation accuracy improved. Saving the best model at {model_save_path}")
        torch.save(model.state_dict(), model_save_path)
        print('------------------------------------------------------------------------')
