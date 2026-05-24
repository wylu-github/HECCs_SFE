import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np
from SF_energy_nn import SFEnergyNN


# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


# Validate the model
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# Hyperparameter tuning function
def hyperparameter_tuning(features, targets, num_epochs=50, output_file="tuning_results.xlsx"):
    input_size = features.shape[1]
    output_size = 1

    hidden_sizes = [8, 16, 32, 64]
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [8, 16, 32, 64]

    # hidden_sizes = [8, 16]
    # learning_rates = [0.BN]
    # batch_sizes = [8, 16]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_loss = float('inf')
    best_params = None

    results = []  # 用于存储超参数及对应损失

    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                fold_losses = []
                for train_idx, val_idx in kf.split(features):
                    train_features, val_features = features[train_idx], features[val_idx]
                    train_targets, val_targets = targets[train_idx], targets[val_idx]

                    train_dataset = TensorDataset(train_features, train_targets)
                    val_dataset = TensorDataset(val_features, val_targets)

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    model = SFEnergyNN(input_size, hidden_size, output_size)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    # Train and validate
                    train_model(model, train_loader, criterion, optimizer, num_epochs)
                    val_loss = validate_model(model, val_loader, criterion)

                    fold_losses.append(val_loss)

                avg_loss = np.mean(fold_losses)
                print(f"Hidden Size: {hidden_size}, Learning Rate: {lr}, Batch Size: {batch_size}, Average Loss: {avg_loss:.4f}")

                # 记录到 results
                results.append([hidden_size, lr, batch_size, avg_loss])

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_params = {'hidden_size': hidden_size, 'lr': lr, 'batch_size': batch_size}

    # 保存到 Excel
    df = pd.DataFrame(results, columns=['Hidden Size', 'Learning Rate', 'Batch Size', 'Average Loss'])
    df.to_excel(output_file, index=False)
    print(f"Tuning results saved to {output_file}")

    print(f"Best Params: {best_params}, Best Loss: {best_loss:.4f}")
    return best_params


if __name__ == "__main__":
    # Load data
    data = pd.read_excel(r'..\get_feature\features_train_4NN.xlsx')
    features = data.iloc[:, 1:-1].values
    targets = data.iloc[:, -1].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    joblib.dump(scaler, 'scaler_perfect.pkl')

    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    # Perform hyperparameter tuning
    best_params = hyperparameter_tuning(features, targets)

    # Save best parameters
    joblib.dump(best_params, "best_params.pkl")
    print("Hyperparameter tuning completed. Best parameters saved.")
