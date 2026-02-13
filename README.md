---
title: Ml Regressor Ui
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
short_description: AI/ML MLP regressor with a User Interface
license: mit
---

# PyTorch Regression Pipeline (End-to-End)

This repo demonstrates an end-to-end PyTorch machine learning workflow for regression using a user interface:
**data loading → train/val/test split → preprocessing (no leakage) → training → validation → checkpointing → final testing**.

## Features
- Proper train/val/test split with reproducible seeding
- Preprocessing fit only on **train** split (prevents leakage)
- PyTorch `DataLoader` pipeline
- MLP regressor baseline
- Early stopping + best-checkpoint saving
- Metrics logged to `runs/<exp_name>/metrics.csv`
- Loss curves saved to `runs/<exp_name>/loss_curve.png`  
- Final test evaluation saved to `runs/<exp_name>/test_summary.json`
- Parity plots saved to  `runs/<exp_name>/parity_xyz.png` (xyz: train, val, test)  

## Dataset
Uses **California Housing** (scikit-learn) and exports it to CSV.  
Total datapoints: 20640  
Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude  
Target Feature: MedHouseVal   
X_shape: (20640, 8)    
y_shape: (20640, 1)

## Model
Uses Multilayer Perceptron, a fully-connected neural network with an input layer, two hidden layers, and an output layer  
Number of neurons:  
  Input layer: 8   
  Number of hidden layers: variable      
  Type of hidden layers: Linear + ReLU + dropout    
  Output layer: 1   (Linear)  
Optimizer: Adam (lr=variable, weight_decay=variable)    
Loss function: Mean Squared Error loss    
 

## How to run   
Input values for the model and data on the left panel   
Press the button to Write baseline.yaml + Run training    
Wait for training to finish   
Observe results including training details, loss curves, and parity plots    

## Folder structure
```
./
├─ README.md                    # This file
├─ requirements.txt
├─ app.py
├─ .gitignore
├─ configs/
│  └─ baseline.yaml             # Contains user-provided data and model details
├─ scripts/
│  └─ download_data.py          # Downloads the required dataset into csv format
├─ data/raw/
│  └─ california_housing.csv    # The downloaded dataset
├─ src/
│  ├─ __init__.py
│  ├─ main.py               
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ make_dataset.py        # Takes the raw data (csv format), separates into (X, y) data
│  │  ├─ preprocess.py          # Scales and transforms the X data
│  │  └─ split.py               # Splits (X, y) into train, validation, and test datasets
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ mlp.py                 # Defines the ML Regressor model 
│  ├─ train/
│  │  ├─ __init__.py
│  │  ├─ engine.py              # Defines training and evaluation functions
│  │  ├─ metrics.py             # Defines the error metrics: Loss, RMSE, MAE, and R^2
│  │  └─ callbacks.py           # Defines early stopping condition
│  └─  utils/
│     ├─ __init__.py
│     ├─ config.py              # Misc functions to load yaml files
│     ├─ io.py                  # Misc functions to save and load json and checkpoint files
│     └─ seed.py                # Ensures reproducibility and deterministic modeling
└─ runs/<exp_name>/             # Results folder
      ├─ checkpoints/       
      │  └─ best.pt             # The best checkpoint stored
      ├─ data_summary.json      # Data summary from the dataset
      ├─ test_summary.json      # Test summary from the model
      ├─ metrics.csv            # Train and validation losses along with validation RMSE, MAE, R^2 logged
      ├─ loss_curve.png         # Train and validation losses plotted
      └─ parity_xyz.png         # Parity plots; xyz: train, val, and test

```
