# ANN Models for Stacking Fault Energy Prediction in HECCs

This repository contains artificial neural network (ANN) models for predicting the stacking fault energies (SFEs) of high-entropy carbide ceramics (HECCs).

Two trained models are provided:

- `stoichiometric`: ANN model for stoichiometric HECCs
- `carbon-deficient`: ANN model for carbon-deficient HECCs

## Repository structure

```text
.
├── stoichiometric/
│   ├── features_train_4NN.xlsx
│   ├── SF_energy_nn.py
│   ├── hyperparameter_tuning.py
│   ├── train_and_evaluate.py
│   ├── best_params.pkl
│   ├── scaler_perfect.pkl
│   ├── final_SF_energy_model.pth
│   ├── tuning_results.xlsx
│   └── model_predictions.xlsx
│
├── carbon-deficient/
│   ├── features_vacancy_train_3NN.xlsx
│   ├── SF_energy_nn.py
│   ├── hyperparameter_tuning.py
│   ├── train_and_evaluate.py
│   ├── best_params.pkl
│   ├── scaler_vacancy.pkl
│   ├── final_SF_energy_model.pth
│   ├── tuning_results.xlsx
│   └── model_predictions.xlsx
│
└── README.md
```

## Description

The ANN models were trained to predict stacking fault energies of HECCs using composition-based descriptors, averaged elemental-property descriptors, and local atomic environment descriptors.

The target property is the stacking fault energy, labeled as `SF_energy` in the dataset.

## Model architecture

Both models use the same fully connected neural network architecture:

```text
Input layer
→ Linear layer
→ ReLU
→ Linear layer
→ ReLU
→ Linear output layer
```

The model is implemented in PyTorch in `SF_energy_nn.py`.

## Datasets

| Folder | Dataset | Number of samples | Descriptor type |
|---|---|---:|---|
| `stoichiometric` | `features_train_4NN.xlsx` | 1800 | 4NN local environment descriptors |
| `carbon-deficient` | `features_vacancy_train_3NN.xlsx` | 1800 | 3NN local environment descriptors |

## Optimized hyperparameters

| Model | Hidden size | Learning rate | Batch size |
|---|---:|---:|---:|
| Stoichiometric | 32 | 0.01 | 8 |
| Carbon-deficient | 8 | 0.001 | 8 |

## Model performance

The dataset was split into 80% training data and 20% test data using a fixed random seed.

| Model | Dataset | R² | MAE | RMSE |
|---|---|---:|---:|---:|
| Stoichiometric | Training | 0.993 | 0.042 | 0.054 |
| Stoichiometric | Test | 0.987 | 0.053 | 0.072 |
| Carbon-deficient | Training | 0.979 | 0.056 | 0.072 |
| Carbon-deficient | Test | 0.964 | 0.074 | 0.097 |

## Usage

Install the required packages:

```bash
pip install torch numpy pandas scikit-learn matplotlib joblib openpyxl
```

To perform hyperparameter tuning:

```bash
python hyperparameter_tuning.py
```

To train and evaluate the final ANN model:

```bash
python train_and_evaluate.py
```

The trained model will be saved as:

```text
final_SF_energy_model.pth
```

The prediction results will be saved as:

```text
model_predictions.xlsx
```

## Notes

When using the trained models for prediction, the input features should be constructed in the same format as the training datasets.

The corresponding scaler should also be used:

| Model | Scaler |
|---|---|
| Stoichiometric | `scaler_perfect.pkl` |
| Carbon-deficient | `scaler_vacancy.pkl` |
