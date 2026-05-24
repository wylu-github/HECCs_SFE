import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from SF_energy_nn import SFEnergyNN
import torch.optim as optim
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False  


def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)

        model.train()

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    return train_losses, test_losses


def evaluate_model(model, train_loader, test_loader):
    model.eval()

    all_train_preds = []
    all_train_labels = []
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            all_train_preds.extend(outputs.cpu().numpy().flatten())  
            all_train_labels.extend(labels.cpu().numpy().flatten())  

        for inputs, labels in test_loader:
            outputs = model(inputs)
            all_test_preds.extend(outputs.cpu().numpy().flatten())  
            all_test_labels.extend(labels.cpu().numpy().flatten()) 

    train_r2 = r2_score(all_train_labels, all_train_preds)
    train_mae = mean_absolute_error(all_train_labels, all_train_preds)
    test_r2 = r2_score(all_test_labels, all_test_preds)
    test_mae = mean_absolute_error(all_test_labels, all_test_preds)

    print(f"Training R²: {train_r2:.4f}, Training MAE: {train_mae:.4f}")
    print(f"Testing R²: {test_r2:.4f}, Testing MAE: {test_mae:.4f}")

    # 保存实际值和预测值到 Excel
    train_results = pd.DataFrame({'Actual': all_train_labels, 'Predicted': all_train_preds})
    test_results = pd.DataFrame({'Actual': all_test_labels, 'Predicted': all_test_preds})

    with pd.ExcelWriter("model_predictions.xlsx") as writer:
        train_results.to_excel(writer, sheet_name='Train', index=False)
        test_results.to_excel(writer, sheet_name='Test', index=False)

    print("Model predictions saved to model_predictions.xlsx")

    return train_r2, train_mae, test_r2, test_mae


def plot_learning_curve(train_losses, test_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()


if __name__ == "__main__":
    SEED = 24
    set_seed(SEED)

    # Load data
    data = pd.read_excel(r'..\get_feature\features_train_4NN.xlsx')
    features = data.iloc[:, 1:-1].values
    targets = data.iloc[:, -1].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    # Split into train and test sets
    train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2, random_state=24)

    train_dataset = TensorDataset(train_features, train_targets)
    test_dataset = TensorDataset(test_features, test_targets)

    best_params = joblib.load("best_params.pkl")
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    # Initialize the model with best parameters
    model = SFEnergyNN(input_size=features.shape[1], hidden_size=best_params['hidden_size'], output_size=1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

    # Train the model and get losses
    print("Training the model...")
    train_losses, test_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=50)

    # Plot learning curve
    plot_learning_curve(train_losses, test_losses)

    # Evaluate the model
    train_r2, train_mae, test_r2, test_mae = evaluate_model(model, train_loader, test_loader)

    # Save the final trained model
    torch.save(model.state_dict(), "final_SF_energy_model.pth")
    print("Final model saved to final_SF_energy_model.pth")

