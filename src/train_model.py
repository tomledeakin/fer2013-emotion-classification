import torch
import torch.nn as nn
from src.models.mlp_model import MLP
from src.models.cnn_model import EmotionCNN
from src.data_processing.load_data import get_data_loaders
from src.utils.utils import compute_accuracy
from tqdm import tqdm


def train_model(config):
    """
    Train a model (MLP or CNN) based on the provided config.

    Args:
        config (dict): Configuration dictionary with model and training details.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load train and validation data
    train_loader, val_loader, _ = get_data_loaders(
        train_dir=config["train_data_path"],
        test_dir=config["val_data_path"],
        label2idx=None,
        batch_size=config["batch_size"]
    )

    # Initialize the model based on the model name in config
    if config["model_name"] == "MLP":
        model = MLP(
            input_dims=config["input_size"] ** 2,
            hidden_dims=config["hidden_dims"],
            output_dims=7
        )
    elif config["model_name"] == "CNN":
        model = CNN(num_classes=7)
    else:
        raise ValueError(f"Unsupported model name: {config['model_name']}")

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training setup
    epochs = config["epochs"]
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_target = []
        train_predict = []

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
        for X_samples, y_samples in train_loader_tqdm:
            X_samples, y_samples = X_samples.to(device), y_samples.to(device)
            optimizer.zero_grad()
            outputs = model(X_samples)
            loss = criterion(outputs, y_samples)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_target.append(y_samples.cpu())
            train_predict.append(outputs.cpu())

        # Calculate training accuracy
        train_target = torch.cat(train_target)
        train_predict = torch.cat(train_predict)
        train_acc = compute_accuracy(train_predict, train_target)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_target = []
        val_predict = []

        with torch.no_grad():
            for X_samples, y_samples in val_loader:
                X_samples, y_samples = X_samples.to(device), y_samples.to(device)
                outputs = model(X_samples)
                loss = criterion(outputs, y_samples)
                val_loss += loss.item()

                val_target.append(y_samples.cpu())
                val_predict.append(outputs.cpu())

        val_target = torch.cat(val_target)
        val_predict = torch.cat(val_predict)
        val_acc = compute_accuracy(val_predict, val_target)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = config["checkpoint_path"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
