# src/modeling/ft_transformer_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from rtdl_revisiting_models import FTTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import time

# --- Global settings / Constants for this module ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Wrapper (from your script) ---
class CatOnlyDataset(Dataset):
    def __init__(self, X_cat_np, y_np): # Expect NumPy arrays
        self.X_cat = torch.as_tensor(X_cat_np, dtype=torch.long)
        self.y     = torch.as_tensor(y_np,     dtype=torch.long)
    def __len__(self):          return self.y.size(0)
    def __getitem__(self, idx): return self.X_cat[idx], self.y[idx]

# --- Focal Loss (from your script) ---
def focal_loss_ft(logits, targets, gamma=2.0, weight=None):
    log_probs = F.log_softmax(logits, dim=1)
    probs     = torch.exp(log_probs)
    tgt_log_p = log_probs[torch.arange(len(targets), device=targets.device), targets]
    tgt_p     = probs[torch.arange(len(targets), device=targets.device), targets]
    loss      = -((1.0 - tgt_p) ** gamma) * tgt_log_p
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()

# --- Epoch Loop ---
def run_epoch_ft(model, loader, loss_fn, optimizer=None, device=DEVICE):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x_cat, y_batch in loader:
        x_cat, y_batch = x_cat.to(device), y_batch.to(device)
        logits   = model(None, x_cat)
        loss     = loss_fn(logits, y_batch)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * y_batch.size(0)
        correct  += (logits.argmax(1) == y_batch).sum().item()
        total    += y_batch.size(0)
    return loss_sum / total, correct / total

# --- Global variables for Optuna objective ---
_X_TRAIN_FOR_OPTUNA_CV = None
_Y_TRAIN_FOR_OPTUNA_CV = None
_CAT_CARDINALITIES_FOR_OPTUNA = None
_N_CLASSES_FOR_OPTUNA = None
_BASE_CLASS_WEIGHTS_FOR_OPTUNA = None
_SEED_FOR_OPTUNA = None
_DEVICE_FOR_OPTUNA = None

def _objective_optuna_ft(trial):
    global _X_TRAIN_FOR_OPTUNA_CV, _Y_TRAIN_FOR_OPTUNA_CV, _CAT_CARDINALITIES_FOR_OPTUNA, \
           _N_CLASSES_FOR_OPTUNA, _BASE_CLASS_WEIGHTS_FOR_OPTUNA, _SEED_FOR_OPTUNA, _DEVICE_FOR_OPTUNA

    d_block          = trial.suggest_categorical("d_block",  [128, 192, 256, 320])
    n_blocks         = trial.suggest_int        ("n_blocks", 2, 6)
    n_heads          = trial.suggest_categorical("attention_n_heads", [4, 8])
    attn_dropout     = trial.suggest_float      ("attention_dropout", 0.0, 0.4)
    ffn_dropout      = trial.suggest_float      ("ffn_dropout",       0.0, 0.4)
    residual_dropout = trial.suggest_float      ("residual_dropout",  0.0, 0.3)
    ffn_mult         = trial.suggest_float      ("ffn_d_hidden_multiplier", 1.0, 6.0)
    lr               = trial.suggest_float      ("lr", 1e-4, 1e-3, log=True)
    batch_size       = trial.suggest_categorical("batch_size", [128, 256, 512])
    loss_type        = trial.suggest_categorical("loss_type", ["ce", "focal"])

    if loss_type == "ce":
        loss_fn = lambda logits, y: F.cross_entropy(logits, y, weight=_BASE_CLASS_WEIGHTS_FOR_OPTUNA, label_smoothing=0.1)
    else:
        loss_fn = lambda logits, y: focal_loss_ft(logits, y, gamma=2.0, weight=_BASE_CLASS_WEIGHTS_FOR_OPTUNA)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=_SEED_FOR_OPTUNA) # Your notebook uses 5
    fold_accs = []

    for tr_idx, va_idx in cv.split(_X_TRAIN_FOR_OPTUNA_CV, _Y_TRAIN_FOR_OPTUNA_CV):
        X_tr_fold, y_tr_fold = _X_TRAIN_FOR_OPTUNA_CV[tr_idx], _Y_TRAIN_FOR_OPTUNA_CV[tr_idx]
        X_va_fold, y_va_fold = _X_TRAIN_FOR_OPTUNA_CV[va_idx], _Y_TRAIN_FOR_OPTUNA_CV[va_idx]
        
        ds_tr = CatOnlyDataset(X_tr_fold, y_tr_fold)
        ds_va = CatOnlyDataset(X_va_fold, y_va_fold)

        class_counts_fold = np.bincount(y_tr_fold)
        if 0 in class_counts_fold or len(class_counts_fold) != _N_CLASSES_FOR_OPTUNA:
             fold_accs.append(0.0) # Penalize if a class is missing
             continue
        samp_wts_fold = 1.0 / class_counts_fold[y_tr_fold]
        sampler_fold = WeightedRandomSampler(weights=samp_wts_fold, num_samples=len(samp_wts_fold), replacement=True)
        
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler_fold)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

        model = FTTransformer(
            n_cont_features=0, cat_cardinalities=_CAT_CARDINALITIES_FOR_OPTUNA, d_out=_N_CLASSES_FOR_OPTUNA,
            n_blocks=n_blocks, d_block=d_block, attention_n_heads=n_heads,
            attention_dropout=attn_dropout, ffn_d_hidden_multiplier=ffn_mult,
            ffn_dropout=ffn_dropout, residual_dropout=residual_dropout,
        ).to(_DEVICE_FOR_OPTUNA)

        opt = torch.optim.AdamW(model.make_parameter_groups(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

        best_fold_val_acc, patience_count, wait_count = 0.0, 10, 0
        for epoch in range(100):
            run_epoch_ft(model, dl_tr, loss_fn, opt, device=_DEVICE_FOR_OPTUNA)
            sched.step()
            _, val_acc_epoch = run_epoch_ft(model, dl_va, loss_fn, device=_DEVICE_FOR_OPTUNA)
            if val_acc_epoch > best_fold_val_acc:
                best_fold_val_acc, wait_count = val_acc_epoch, 0
            else:
                wait_count += 1
            if wait_count >= patience_count:
                break
        fold_accs.append(best_fold_val_acc)
    
    return float(np.mean(fold_accs)) if fold_accs else 0.0


def train_and_evaluate_ft_transformer(
    X_train_cat_np, y_train_np, X_test_cat_np, y_test_np,
    cat_cardinalities_list, num_target_classes, feature_names_list, target_encoder_classes,
    seed=42,
    optuna_n_trials=16,
    optuna_timeout_hours=1,
    final_model_epochs=50
    ):
    """
    Main function to run the FT-Transformer experiment: tunes with Optuna,
    trains the final model, and evaluates.
    Returns the trained model and its predictions on the test set.
    """
    global _X_TRAIN_FOR_OPTUNA_CV, _Y_TRAIN_FOR_OPTUNA_CV, _CAT_CARDINALITIES_FOR_OPTUNA, \
           _N_CLASSES_FOR_OPTUNA, _BASE_CLASS_WEIGHTS_FOR_OPTUNA, _SEED_FOR_OPTUNA, _DEVICE_FOR_OPTUNA

    print("\n" + "---" * 10 + " FT-Transformer Experiment " + "---" * 10)
    start_exp_time = time.time()

    # Set seeds
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    _X_TRAIN_FOR_OPTUNA_CV = X_train_cat_np
    _Y_TRAIN_FOR_OPTUNA_CV = y_train_np
    _CAT_CARDINALITIES_FOR_OPTUNA = cat_cardinalities_list
    _N_CLASSES_FOR_OPTUNA = num_target_classes
    _SEED_FOR_OPTUNA = seed
    _DEVICE_FOR_OPTUNA = DEVICE
    
    _BASE_CLASS_WEIGHTS_FOR_OPTUNA = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(y_train_np), y=y_train_np),
        dtype=torch.float32, device=DEVICE
    )

    print("\n--- Starting Optuna Study for FT-Transformer ---")
    study_start_time = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective_optuna_ft, n_trials=optuna_n_trials, timeout=optuna_timeout_hours*60*60)
    print(f"Optuna study took {(time.time() - study_start_time)/60:.2f} minutes.")

    print("\nOptuna study complete.")
    print("Best mean-CV accuracy from Optuna:", study.best_value)
    best_hyperparams = study.best_params
    print("Best hyper-parameters from Optuna:", best_hyperparams)

    # --- Train Final Model with Best Hyperparameters ---
    print("\n--- Training Final FT-Transformer model with best parameters ---")
    
    if best_hyperparams["loss_type"] == "ce":
        final_loss_fn = lambda l, y: F.cross_entropy(l, y, weight=_BASE_CLASS_WEIGHTS_FOR_OPTUNA, label_smoothing=0.1)
    else:
        final_loss_fn = lambda l, y: focal_loss_ft(l, y, gamma=2.0, weight=_BASE_CLASS_WEIGHTS_FOR_OPTUNA)

    final_model = FTTransformer(
        n_cont_features=0, cat_cardinalities=cat_cardinalities_list, d_out=num_target_classes,
        n_blocks=best_hyperparams["n_blocks"], d_block=best_hyperparams["d_block"],
        attention_n_heads=best_hyperparams["attention_n_heads"],
        attention_dropout=best_hyperparams["attention_dropout"],
        ffn_d_hidden_multiplier=best_hyperparams["ffn_d_hidden_multiplier"],
        ffn_dropout=best_hyperparams["ffn_dropout"],
        residual_dropout=best_hyperparams["residual_dropout"],
    ).to(DEVICE)

    final_opt = torch.optim.AdamW(final_model.make_parameter_groups(), lr=best_hyperparams["lr"], weight_decay=1e-2)
    final_sched = torch.optim.lr_scheduler.CosineAnnealingLR(final_opt, T_max=20)

    counts_final_train = np.bincount(y_train_np)
    weights_final_train = 1.0 / counts_final_train[y_train_np]
    sampler_final_train = WeightedRandomSampler(weights_final_train, len(weights_final_train), replacement=True)
    
    final_train_dataset = CatOnlyDataset(X_train_cat_np, y_train_np)
    final_train_loader = DataLoader(final_train_dataset,
                                    batch_size=best_hyperparams["batch_size"], 
                                    sampler=sampler_final_train)
    
    final_model_train_start_time = time.time()
    for epoch in range(final_model_epochs):
        if (epoch + 1) % 10 == 0 or epoch == 0:
             print(f"Final FT-Transformer Training Epoch: {epoch+1}/{final_model_epochs}")
        run_epoch_ft(final_model, final_train_loader, final_loss_fn, final_opt, device=DEVICE)
        final_sched.step()
    
    print(f"Final FT-Transformer training done in {(time.time() - final_model_train_start_time):.2f} seconds.")
    final_model.eval()

    # --- Evaluation on Test Set ---
    print("\n--- Evaluating Final FT-Transformer on Test Set ---")
    test_dataset = CatOnlyDataset(X_test_cat_np, y_test_np)
    test_loader = DataLoader(test_dataset, batch_size=best_hyperparams["batch_size"], shuffle=False)
    
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x_cat_batch, y_batch in test_loader:
            logits_batch = final_model(None, x_cat_batch.to(DEVICE))
            all_logits.append(logits_batch.cpu())
            all_labels.append(y_batch)
            
    logits_np = torch.cat(all_logits).numpy()
    y_true_np = torch.cat(all_labels).numpy()
    y_pred_np = logits_np.argmax(1)

    # --- Permutation Importance ---
    exp_time = time.time() - start_exp_time
    print(f"FT-Transformer Experiment took {exp_time/60:.2f} minutes.")
    print("---" * 20)
    return final_model, y_true_np, y_pred_np, best_hyperparams
