# src/modeling/basic_nn_models.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # WeightedRandomSampler if using your oversample
import numpy as np
import pandas as pd # For oversample if kept here
from sklearn.utils import resample # For oversample
from sklearn.metrics import f1_score # For validation metric during tuning
# from collections import Counter # For class weights if needed directly
import time
import random # For shuffling search space

# SEED should ideally be passed or set globally by main.py
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- PyTorch Dataset (from your script) ---
class CatDatasetNN(Dataset): # Renamed to avoid conflict
    def __init__(self, X_np, y_np): # Expects NumPy arrays
        self.X = torch.tensor(X_np, dtype=torch.long)
        self.y = torch.tensor(y_np, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# --- Deep Embedding Network (from your script, slightly adapted) ---
class EmbeddingNN(nn.Module):
    def __init__(self, cat_dims_list, num_target_classes, hidden_layers_config, dropout_rate, embedding_dim_rule="default"):
        super().__init__()
        
        # Embedding dimensions
        if embedding_dim_rule == "default": # Your min(64, (d+1)//2) rule
            emb_dims_list   = [min(64, (d+1)//2) for d in cat_dims_list]
        elif isinstance(embedding_dim_rule, int): # Fixed embedding dimension
            emb_dims_list = [embedding_dim_rule] * len(cat_dims_list)
        else: # Pass a list of embedding dimensions
            emb_dims_list = embedding_dim_rule

        self.embeddings  = nn.ModuleList([nn.Embedding(d, e) for d, e in zip(cat_dims_list, emb_dims_list)])
        self.embedding_dropout = nn.Dropout(dropout_rate) # Dropout after embeddings often helps
        
        total_embedding_dim = sum(emb_dims_list)
        self.batch_norm_embeddings = nn.BatchNorm1d(total_embedding_dim)
        
        layer_list, current_dim = [], total_embedding_dim
        for hidden_units in hidden_layers_config:
            layer_list.extend([
                nn.Linear(current_dim, hidden_units), 
                nn.ReLU(), 
                nn.BatchNorm1d(hidden_units), # Batch norm after activation often used
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_units
        layer_list.append(nn.Linear(current_dim, num_target_classes))
        self.mlp_layers = nn.Sequential(*layer_list)

    def forward(self, x_cat): # x_cat is (batch_size, num_features)
        z = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        z = torch.cat(z, 1)
        z = self.embedding_dropout(z) # Apply dropout to embeddings
        z = self.batch_norm_embeddings(z)
        return self.mlp_layers(z)

# --- Manual Oversample function (from your script) ---
# This should ideally be in src/utils.py but including here for self-containment of this module for now
def manual_oversample_nn(X_df, y_series, random_state=42):
    df_combined = pd.concat([X_df.reset_index(drop=True), y_series.reset_index(drop=True).rename("TARGET_CLASS")], axis=1)
    max_class_count = df_combined["TARGET_CLASS"].value_counts().max()
    
    df_oversampled_list = []
    for class_val in df_combined["TARGET_CLASS"].unique():
        df_class = df_combined[df_combined["TARGET_CLASS"] == class_val]
        df_oversampled_list.append(resample(df_class, 
                                             replace=True, 
                                             n_samples=max_class_count, 
                                             random_state=random_state))
    df_oversampled = pd.concat(df_oversampled_list)
    return df_oversampled.drop(columns="TARGET_CLASS"), df_oversampled["TARGET_CLASS"]


def train_and_evaluate_embedding_nn(
    X_train_np, y_train_np, # LabelEncoded NumPy arrays
    X_val_np, y_val_np,
    X_test_np, y_test_np,
    cat_dims_list,
    num_target_classes,
    target_encoder_classes, # For classification report
    original_feature_names, # For feature importance (if applicable, permutation)
    model_type_name="EmbeddingNN", # e.g. "MLP" or "EE-NN" for printouts
    seed=42,
    # Hyperparameters for the search or fixed run
    hyperparam_configs_list=None, # List of dicts for hyperparameter trials
    max_search_trials=10,
    epochs_per_trial=15, # Default epochs for each config
    patience_early_stopping=4, # Default patience
    batch_size_nn=64,
    use_manual_oversampling=True, # Flag from config
    use_class_weights_loss=True  # Flag from config
    ):
    """
    Performs hyperparameter search (if configs provided) or trains a single config
    for EmbeddingNN (MLP/EE-NN like).
    Returns the best trained model and its predictions on the test set.
    """
    print("\n" + "---" * 10 + f" {model_type_name} Experiment " + "---" * 10)
    exp_start_time = time.time()

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)

    # --- Data Handling ---
    # Convert input NumPy arrays to DataFrames if manual_oversample needs them
    X_train_df_for_os = pd.DataFrame(X_train_np, columns=[f"f{i}" for i in range(X_train_np.shape[1])]) # Generic col names
    y_train_series_for_os = pd.Series(y_train_np)

    if use_manual_oversampling:
        print("Applying manual oversampling to training data...")
        X_train_processed_df, y_train_processed_series = manual_oversample_nn(X_train_df_for_os, y_train_series_for_os, random_state=seed)
        X_train_processed_np = X_train_processed_df.values
        y_train_processed_np = y_train_processed_series.values
        print(f"Oversampled training data shape: X={X_train_processed_np.shape}, y={y_train_processed_np.shape}")
    else:
        X_train_processed_np = X_train_np
        y_train_processed_np = y_train_np

    train_dataset = CatDatasetNN(X_train_processed_np, y_train_processed_np)
    val_dataset = CatDatasetNN(X_val_np, y_val_np)
    test_dataset = CatDatasetNN(X_test_np, y_test_np)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_nn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_nn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_nn, shuffle=False)

    # Class weights for loss function (calculated on potentially oversampled training data)
    loss_class_weights = None
    if use_class_weights_loss:
        class_counts = np.bincount(y_train_processed_np)
        total_samples = sum(class_counts)
        loss_class_weights = torch.tensor([total_samples / (len(class_counts) * count) if count > 0 else 0 
                                           for count in class_counts], dtype=torch.float32).to(DEVICE)
        print(f"Using class weights for loss: {loss_class_weights}")


    # --- Hyperparameter Search or Single Run ---
    if hyperparam_configs_list is None or not self.config['run_gridsearch']: # Assuming config has run_gridsearch
        # Use default/fixed params if no search list provided or gridsearch is off
        # These defaults should come from your config for MLP/EE-NN baseline
        print("Running with fixed/default hyperparameters (no extensive search).")
        default_params = { # Example defaults, should come from config
            'hidden_layers': [128, 64], 'dropout': 0.3, 'lr': 0.001,
            'embedding_rule': 'default' # or a fixed int
        }
        hyperparam_configs_list = [default_params]
        max_search_trials = 1 # Just run one config

    best_val_metric = -1 # Using Macro F1 for validation
    best_model_state_dict = None
    best_run_config = None

    print(f"\n--- Starting Hyperparameter Search/Run for {model_type_name} (max {max_search_trials} trials) ---")
    search_start_time = time.time()

    for i, current_config in enumerate(hyperparam_configs_list[:max_search_trials]):
        print(f"\n🔍 Trial {i+1}/{max_search_trials}: {current_config}")
        trial_start_time = time.time()

        model_instance = EmbeddingNN(
            cat_dims_list=cat_dims_list,
            num_target_classes=num_target_classes,
            hidden_layers_config=current_config['hidden_layers'],
            dropout_rate=current_config['dropout'],
            embedding_dim_rule=current_config.get('embedding_rule', 'default') # Or pass fixed int
        ).to(DEVICE)

        # Loss function (Using CrossEntropy, FocalLoss can be added as a config option)
        # Your MLP/EE-NN used Focal Loss with inverse class frequency.
        # Let's use CrossEntropy with class_weights for now, similar to your script.
        loss_fn_instance = nn.CrossEntropyLoss(weight=loss_class_weights if use_class_weights_loss else None)
        optimizer_instance = torch.optim.AdamW(model_instance.parameters(), lr=current_config['lr'])
        # scheduler_instance = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_instance, T_max=epochs_per_trial)

        current_best_val_trial_metric = -1
        patience_counter = 0

        for epoch in range(epochs_per_trial):
            model_instance.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer_instance.zero_grad()
                outputs = model_instance(X_batch)
                loss = loss_fn_instance(outputs, y_batch)
                loss.backward()
                optimizer_instance.step()
            # scheduler_instance.step() # If using scheduler

            # Validation
            model_instance.eval()
            val_preds_epoch, val_true_epoch = [], []
            with torch.no_grad():
                for X_batch_val, y_batch_val in val_loader:
                    X_batch_val, y_batch_val = X_batch_val.to(DEVICE), y_batch_val.to(DEVICE)
                    outputs_val = model_instance(X_batch_val)
                    val_preds_epoch.extend(outputs_val.argmax(1).cpu().numpy())
                    val_true_epoch.extend(y_batch_val.cpu().numpy())
            
            current_val_metric_epoch = f1_score(val_true_epoch, val_preds_epoch, average='macro', zero_division=0)
            if (epoch + 1) % 5 == 0 or epoch == 0 or epochs_per_trial == 1: # Print every 5 epochs
                 print(f"Epoch {epoch+1}/{epochs_per_trial} - Val Macro F1: {current_val_metric_epoch:.4f}")

            if current_val_metric_epoch > current_best_val_trial_metric:
                current_best_val_trial_metric = current_val_metric_epoch
                best_model_state_dict_trial = deepcopy(model_instance.state_dict()) # Deepcopy for safety
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience_early_stopping:
                print(f"Early stopping at epoch {epoch+1} for trial {i+1}.")
                break
        
        print(f"Trial {i+1} Best Val Macro F1: {current_best_val_trial_metric:.4f} (took {time.time() - trial_start_time:.2f}s)")
        if current_best_val_trial_metric > best_val_metric:
            best_val_metric = current_best_val_trial_metric
            best_model_state_dict = best_model_state_dict_trial
            best_run_config = current_config

    print(f"{model_type_name} hyperparameter search/run took {(time.time() - search_start_time)/60:.2f} minutes.")
    if best_run_config:
        print(f"\n✅ Best {model_type_name} Config found:", best_run_config)
        print(f"🏆 Best {model_type_name} Validation Macro F1 Score: {best_val_metric:.4f}")
    else:
        print(f"No successful trials for {model_type_name}. Cannot determine best config.")
        return None, None, None, None # Or raise an error

    # --- Instantiate final model with best state dict ---
    final_model_instance = EmbeddingNN(
        cat_dims_list=cat_dims_list,
        num_target_classes=num_target_classes,
        hidden_layers_config=best_run_config['hidden_layers'],
        dropout_rate=best_run_config['dropout'],
        embedding_dim_rule=best_run_config.get('embedding_rule', 'default')
    ).to(DEVICE)
    final_model_instance.load_state_dict(best_model_state_dict)
    final_model_instance.eval()

    # --- Evaluation on Test Set ---
    print(f"\n--- Evaluating Final {model_type_name} on Test Set ---")
    test_preds_final, test_true_final = [], []
    with torch.no_grad():
        for X_batch_test, y_batch_test in test_loader:
            X_batch_test = X_batch_test.to(DEVICE)
            outputs_test = final_model_instance(X_batch_test)
            test_preds_final.extend(outputs_test.argmax(1).cpu().numpy())
            test_true_final.extend(y_batch_test.cpu().numpy()) # y_batch_test is already on CPU

    exp_end_time = time.time()
    print(f"{model_type_name} Experiment (including tuning/final eval) took {(exp_end_time - exp_start_time)/60:.2f} minutes.")
    print("---" * 20)
    return final_model_instance, np.array(test_true_final), np.array(test_preds_final), best_run_config