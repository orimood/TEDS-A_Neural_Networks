# src/modeling/basic_nn_models.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # WeightedRandomSampler if using your oversample
import numpy as np
import pandas as pd # For oversample function
from sklearn.utils import resample # For oversample
from sklearn.metrics import f1_score # For validation metric during tuning
# from collections import Counter # Not strictly needed if using np.bincount
import time
import random # For shuffling search space
from copy import deepcopy # For model state dict

# Global device setting
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- PyTorch Dataset ---
class CatDatasetForNN(Dataset):
    def __init__(self, X_np, y_np): # Expects NumPy arrays
        self.X = torch.tensor(X_np, dtype=torch.long)
        self.y = torch.tensor(y_np, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# --- Embedding Neural Network ---
class EmbeddingNN(nn.Module):
    def __init__(self, cat_dims_list, num_target_classes, hidden_layers_config, dropout_rate, 
                 embedding_dim_rule="default_rule"):
        super().__init__()
        
        if not cat_dims_list:
            raise ValueError("cat_dims_list cannot be empty for EmbeddingNN.")

        if embedding_dim_rule == "default_rule":
            emb_dims_list = [min(50, (d + 1) // 2) for d in cat_dims_list] # Adjusted min to 50 from 64
        elif isinstance(embedding_dim_rule, int):
            emb_dims_list = [embedding_dim_rule] * len(cat_dims_list)
        elif isinstance(embedding_dim_rule, list) and len(embedding_dim_rule) == len(cat_dims_list):
            emb_dims_list = embedding_dim_rule
        else:
            print(f"Warning: Invalid embedding_dim_rule ('{embedding_dim_rule}'). Falling back to default.")
            emb_dims_list = [min(50, (d + 1) // 2) for d in cat_dims_list]

        self.embeddings  = nn.ModuleList([nn.Embedding(num_categories, emb_dim) 
                                          for num_categories, emb_dim in zip(cat_dims_list, emb_dims_list)])
        
        total_embedding_dim = sum(emb_dims_list)
        if total_embedding_dim == 0: # Should not happen if cat_dims_list is not empty
             raise ValueError("Total embedding dimension is zero. Check cat_dims_list and embedding_dim_rule.")

        self.embedding_batch_norm = nn.BatchNorm1d(total_embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        layer_list, current_dim = [], total_embedding_dim
        for hidden_units in hidden_layers_config:
            layer_list.extend([
                nn.Linear(current_dim, hidden_units), 
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_units), 
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_units
        layer_list.append(nn.Linear(current_dim, num_target_classes))
        self.mlp_layers = nn.Sequential(*layer_list)

    def forward(self, x_cat):
        if x_cat.shape[1] != len(self.embeddings):
            raise ValueError(f"Input x_cat has {x_cat.shape[1]} features, but model expects {len(self.embeddings)} based on embeddings.")
        z = [self.embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])]
        z = torch.cat(z, 1)
        z = self.embedding_batch_norm(z)
        z = self.embedding_dropout(z)
        return self.mlp_layers(z)

# --- Manual Oversample function ---
def _manual_oversample_for_nn(X_df_input, y_series_input, random_state=42):
    df_combined = pd.concat([X_df_input.reset_index(drop=True), 
                             y_series_input.reset_index(drop=True).rename("TARGET_CLASS")], axis=1)
    max_class_count = df_combined["TARGET_CLASS"].value_counts().max()
    if pd.isna(max_class_count): # Handle empty or all NaN y_series_input
        print("Warning: Could not determine max class count for oversampling. Returning original data.")
        return X_df_input, y_series_input

    df_oversampled_list = []
    for class_val in df_combined["TARGET_CLASS"].unique():
        df_class = df_combined[df_combined["TARGET_CLASS"] == class_val]
        if not df_class.empty:
            df_oversampled_list.append(resample(df_class, 
                                                 replace=True, 
                                                 n_samples=int(max_class_count), # Ensure n_samples is int 
                                                 random_state=random_state))
    if not df_oversampled_list: # Handle case where all classes were empty
        print("Warning: No data to oversample. Returning original data.")
        return X_df_input, y_series_input
        
    df_oversampled = pd.concat(df_oversampled_list)
    return df_oversampled.drop(columns="TARGET_CLASS"), df_oversampled["TARGET_CLASS"]


def train_and_evaluate_embedding_nn(
    X_train_np, y_train_np, 
    X_val_np, y_val_np,
    X_test_np, y_test_np,
    cat_dims_list,          
    num_target_classes,
    original_feature_names, # Used if converting X_train_np to df for oversampling
    model_type_name="EmbeddingNN", 
    seed=42,
    hyperparam_configs_list=None, 
    max_search_trials=1, # Default to 1 if hyperparam_configs_list is for a single run
    epochs_per_trial=15,        
    patience_early_stop=4,    
    batch_size=64,           
    use_manual_oversampling=True, 
    use_class_weights_loss=True,
    run_hyperparam_search=False # Explicit flag to control search
    ):
    """
    Trains/tunes and evaluates an EmbeddingNN (MLP/EE-NN like).
    """
    print("\n" + "---" * 10 + f" {model_type_name} Experiment " + "---" * 10)
    exp_start_time = time.time()

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)

    if use_manual_oversampling:
        print("Applying manual oversampling to training data for NN...")
        X_train_df_for_os = pd.DataFrame(X_train_np, columns=original_feature_names)
        y_train_series_for_os = pd.Series(y_train_np)
        X_train_os_df, y_train_os_series = _manual_oversample_for_nn(
            X_train_df_for_os, y_train_series_for_os, random_state=seed
        )
        X_train_feed_np = X_train_os_df.values
        y_train_feed_np = y_train_os_series.values
        print(f"Oversampled training data shape for NN: X={X_train_feed_np.shape}, y={y_train_feed_np.shape}")
    else:
        X_train_feed_np = X_train_np
        y_train_feed_np = y_train_np

    train_dataset = CatDatasetForNN(X_train_feed_np, y_train_feed_np)
    val_dataset = CatDatasetForNN(X_val_np, y_val_np)
    test_dataset = CatDatasetForNN(X_test_np, y_test_np)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    loss_class_weights_tensor = None
    if use_class_weights_loss:
        class_counts_train = np.bincount(y_train_feed_np, minlength=num_target_classes)
        # Ensure no division by zero for classes not present in y_train_feed_np (after oversampling this is less likely for all classes)
        if not np.all(class_counts_train > 0):
            print(f"Warning: Some classes have zero samples in y_train_feed_np. Counts: {class_counts_train}")
            # Fallback or adjust weights carefully. For simplicity, proceeding but this could be an issue.
        
        total_samples_train = sum(class_counts_train)
        if total_samples_train > 0:
            loss_class_weights_tensor = torch.tensor(
                [total_samples_train / (len(class_counts_train) * count) if count > 0 else 1.0 # Assign weight 1 if count is 0
                 for count in class_counts_train], dtype=torch.float32).to(DEVICE)
            print(f"Using class weights for NN loss: {loss_class_weights_tensor}")
        else:
            print("Warning: Total samples in y_train_feed_np is zero. Not using class weights.")

    if not run_hyperparam_search or not hyperparam_configs_list:
        print(f"Running {model_type_name} with a single/default hyperparameter configuration.")
        # Define your single "best" or default config for MLP and EE-NN from your images/notes
        # This should be passed in hyperparam_configs_list as a list with one element if not searching
        if hyperparam_configs_list and isinstance(hyperparam_configs_list, list) and len(hyperparam_configs_list) == 1:
             current_configs_to_try = hyperparam_configs_list
        else: # Fallback to some very basic defaults
            print("Warning: No specific single config provided and search is off. Using basic defaults.")
            current_configs_to_try = [{ 
                'hidden_layers': [128, 64], 'dropout': 0.3, 'lr': 0.001,
                'embedding_rule': 'default_rule'
            }]
        num_trials_to_run = 1
    else:
        current_configs_to_try = hyperparam_configs_list
        num_trials_to_run = min(max_search_trials, len(current_configs_to_try))
        if num_trials_to_run < len(current_configs_to_try):
            print(f"Note: Trying {num_trials_to_run} random configurations from the provided list of {len(current_configs_to_try)}.")
            random.shuffle(current_configs_to_try) # Shuffle if trying a subset
            current_configs_to_try = current_configs_to_try[:num_trials_to_run]


    best_val_metric_overall = -1.0
    best_model_state_final = None
    best_config_final = None
    
    print(f"\n--- Starting Hyperparameter Trials for {model_type_name} (up to {num_trials_to_run} trials) ---")
    search_start_time = time.time()

    for i, current_trial_config in enumerate(current_configs_to_try):
        print(f"\n🔍 NN Trial {i+1}/{num_trials_to_run}: {current_trial_config}")
        trial_run_start_time = time.time()

        model = EmbeddingNN(
            cat_dims_list=cat_dims_list,
            num_target_classes=num_target_classes,
            hidden_layers_config=current_trial_config['hidden_layers'],
            dropout_rate=current_trial_config['dropout'],
            embedding_dim_rule=current_trial_config.get('embedding_rule', 'default_rule')
        ).to(DEVICE)

        loss_function = nn.CrossEntropyLoss(weight=loss_class_weights_tensor if use_class_weights_loss else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=current_trial_config['lr'])
        
        current_best_val_metric_trial = -1.0
        trial_best_state_dict = None
        patience_current_trial = 0

        for epoch in range(epochs_per_trial):
            model.train()
            train_loss_epoch = 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_b)
                loss = loss_function(outputs, y_b)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
            
            model.eval()
            val_preds_list, val_true_list = [], []
            with torch.no_grad():
                for X_b_val, y_b_val in val_loader:
                    X_b_val = X_b_val.to(DEVICE)
                    outputs_val = model(X_b_val)
                    val_preds_list.extend(outputs_val.argmax(1).cpu().numpy())
                    val_true_list.extend(y_b_val.cpu().numpy()) # y_b_val is already on CPU
            
            epoch_val_metric = f1_score(val_true_list, val_preds_list, average='macro', zero_division=0)
            if (epoch + 1) % 5 == 0 or epoch == 0 or epochs_per_trial == 1:
                 print(f"Epoch {epoch+1}/{epochs_per_trial} - Val Macro F1: {epoch_val_metric:.4f} - Train Loss: {train_loss_epoch/len(train_loader):.4f}")

            if epoch_val_metric > current_best_val_metric_trial:
                current_best_val_metric_trial = epoch_val_metric
                trial_best_state_dict = deepcopy(model.state_dict())
                patience_current_trial = 0
            else:
                patience_current_trial += 1
            
            if patience_current_trial >= patience_early_stop:
                print(f"Early stopping at epoch {epoch+1} for trial {i+1}.")
                break
        
        print(f"Trial {i+1} completed. Best Val Macro F1 for this trial: {current_best_val_metric_trial:.4f}. Took {time.time() - trial_run_start_time:.2f}s")
        if current_best_val_metric_trial > best_val_metric_overall:
            best_val_metric_overall = current_best_val_metric_trial
            best_model_state_final = trial_best_state_dict
            best_config_final = current_trial_config
            print(f"*** New overall best {model_type_name} validation Macro F1: {best_val_metric_overall:.4f} with config: {best_config_final} ***")

    print(f"\n{model_type_name} hyperparameter search/run took {(time.time() - search_start_time)/60:.2f} minutes.")
    if not best_config_final:
        print(f"Error: No successful trials for {model_type_name} or best_config_final not set. Cannot proceed.")
        return None, None, None, None

    final_model = EmbeddingNN( # Renamed from final_model_instance
        cat_dims_list=cat_dims_list,
        num_target_classes=num_target_classes,
        hidden_layers_config=best_config_final['hidden_layers'],
        dropout_rate=best_config_final['dropout'],
        embedding_dim_rule=best_config_final.get('embedding_rule', 'default_rule')
    ).to(DEVICE)
    final_model.load_state_dict(best_model_state_final)
    final_model.eval()

    print(f"\n--- Evaluating Final {model_type_name} on Test Set ---")
    test_preds_final_list, test_true_final_list = [], []
    with torch.no_grad():
        for X_b_test, y_b_test in test_loader:
            X_b_test = X_b_test.to(DEVICE)
            outputs_test = final_model(X_b_test)
            test_preds_final_list.extend(outputs_test.argmax(1).cpu().numpy())
            test_true_final_list.extend(y_b_test.cpu().numpy())

    y_true_test_final_np = np.array(test_true_final_list)
    y_pred_test_final_np = np.array(test_preds_final_list)

    exp_end_time = time.time()
    print(f"{model_type_name} Experiment main part took {(exp_end_time - exp_start_time)/60:.2f} minutes.")
    print("---" * 20)
    return final_model, y_true_test_final_np, y_pred_test_final_np, best_config_final