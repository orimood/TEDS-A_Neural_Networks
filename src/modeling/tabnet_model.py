# src/modeling/tabnet_model.py

import pandas as pd
import numpy as np
import random
import torch # TabNet is PyTorch based
from sklearn.preprocessing import LabelEncoder # Assuming features are already label encoded by preprocess_teds_data
from sklearn.metrics import f1_score, classification_report, accuracy_score
# from sklearn.utils import resample # manual_oversample will be a utility if needed
from pytorch_tabnet.tab_model import TabNetClassifier
import time

# SEED should ideally be passed or set globally by main.py
# For this example, let's assume it's passed as an argument where needed.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_and_evaluate_tabnet(
    X_train_np, y_train_np, # These are already label encoded numpy arrays
    X_val_np, y_val_np,
    X_test_np, y_test_np,
    cat_idxs_list,          # List of indices of categorical features (all of them)
    cat_dims_list,          # List of cardinalities for each categorical feature
    target_encoder_classes, # For printing classification report with original labels
    seed=42,
    # TabNet specific search parameters from your notebook can be passed via a config
    search_space_configs=None, # List of dicts for hyperparameter trials
    max_search_trials=10,      # Limit number of trials from search_space
    tabnet_default_n_steps=5,
    tabnet_default_cat_emb_dim=1,
    max_epochs_per_trial=100,
    patience_per_trial=10,
    batch_size_tabnet=64, # From your notebook
    virtual_batch_size_tabnet=32 # From your notebook
    ):
    """
    Tunes and trains TabNet, then evaluates on the test set.
    Returns the best trained TabNet model and its predictions on the test set.
    """
    print("\n" + "---" * 10 + " TabNet Experiment " + "---" * 10)
    exp_start_time = time.time()

    # Ensure reproducibility for this run
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if search_space_configs is None: # Default search space if not provided
        search_space_configs = [
            {"n_d": d, "n_a": a, "lr": lr, "gamma": g}
            for d in [32, 64]
            for a in [32, 64]
            for lr in [1e-3, 5e-4]
            for g in [1.0, 1.5, 2.0] # Added 1.5 for gamma
        ]
        random.shuffle(search_space_configs) # Shuffle to get varied configs if max_search_trials is low

    best_macro_f1_val = -1
    best_tabnet_model_state = None
    best_tabnet_config = None

    print(f"\n--- Starting Hyperparameter Search for TabNet (max {max_search_trials} trials) ---")
    search_start_time = time.time()

    for i, params_config in enumerate(search_space_configs[:max_search_trials]):
        print(f"\n🔍 TabNet Trial {i+1}/{max_search_trials}: {params_config}")
        trial_start_time = time.time()
        
        current_model = TabNetClassifier(
            n_d=params_config.get("n_d", 8), # Default TabNet value if not in search
            n_a=params_config.get("n_a", 8), # Default TabNet value if not in search
            gamma=params_config.get("gamma", 1.3), # Default TabNet value
            n_steps=tabnet_default_n_steps,
            verbose=0, # TabNet's own verbosity
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=params_config.get("lr", 2e-2)), # Default TabNet lr is 0.02
            cat_idxs=cat_idxs_list,
            cat_dims=cat_dims_list,
            cat_emb_dim=tabnet_default_cat_emb_dim,
            device_name=DEVICE,
            seed=seed
        )

        current_model.fit(
            X_train=X_train_np, y_train=y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=["val"],
            eval_metric=["balanced_accuracy"], # Or 'macro_f1' if TabNet supports it directly for early stopping
            max_epochs=max_epochs_per_trial,
            patience=patience_per_trial,
            batch_size=batch_size_tabnet,
            virtual_batch_size=virtual_batch_size_tabnet,
            num_workers=0, # To avoid potential issues in some environments
            drop_last=False # Usually False for classification
        )
        
        # Evaluate on validation set for this trial
        val_preds = current_model.predict(X_val_np)
        macro_f1_val_trial = f1_score(y_val_np, val_preds, average="macro", zero_division=0)
        print(f"Trial {i+1} Val Macro F1: {macro_f1_val_trial:.4f} (took {time.time() - trial_start_time:.2f}s)")

        if macro_f1_val_trial > best_macro_f1_val:
            best_macro_f1_val = macro_f1_val_trial
            best_tabnet_model_state = current_model.network.state_dict() # Save state_dict
            best_tabnet_config = params_config
            # Save the model object itself if needed, or just its parameters for re-instantiation
            # best_tabnet_full_model_object = current_model
            print(f"New best TabNet validation Macro F1: {best_macro_f1_val:.4f}")

    print(f"TabNet hyperparameter search took {(time.time() - search_start_time)/60:.2f} minutes.")
    print("\n✅ Best TabNet Hyperparameters found:", best_tabnet_config)
    print(f"🏆 Best TabNet Validation Macro F1 Score: {best_macro_f1_val:.4f}")

    # --- Train Final Model with Best Hyperparameters on full X_train_np + X_val_np ---
    # (Or simply use the best model object if early stopping was on validation set)
    # For TabNet, it's common to just use the model from the best trial if eval_set was used for early stopping
    # If you want to retrain on combined train+val:
    # X_train_full_np = np.concatenate((X_train_np, X_val_np))
    # y_train_full_np = np.concatenate((y_train_np, y_val_np))
    
    print("\n--- Using best TabNet model from tuning for final evaluation ---")
    # Re-instantiate and load best state to ensure clean model
    final_tabnet_model = TabNetClassifier(
        n_d=best_tabnet_config.get("n_d", 8),
        n_a=best_tabnet_config.get("n_a", 8),
        gamma=best_tabnet_config.get("gamma", 1.3),
        n_steps=tabnet_default_n_steps,
        verbose=0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=best_tabnet_config.get("lr", 2e-2)),
        cat_idxs=cat_idxs_list,
        cat_dims=cat_dims_list,
        cat_emb_dim=tabnet_default_cat_emb_dim,
        device_name=DEVICE,
        seed=seed
    )
    # To load from state_dict, model needs to be initialized first.
    # Then, if you just saved the TabNetClassifier object from the best trial, you can use it.
    # If you saved state_dict:
    # final_tabnet_model.load_model(path_to_best_model_zip) # TabNet saves as a zip
    # Or, if you have the model object:
    # final_tabnet_model = best_tabnet_full_model_object # if you stored the model object
    
    # For simplicity, if your search loop already uses X_val for early stopping,
    # the 'current_model' when best_macro_f1_val is updated is effectively your best tuned model.
    # Let's assume we need to re-instantiate and train for a few epochs or load state.
    # Given the way TabNet's `fit` works with `eval_set`, the model object `current_model` at the point
    # it achieved `best_macro_f1_val` is already trained.
    # For this refactor, let's just re-create it and re-train it on the full training data for a few epochs
    # as a demonstration, though using the saved state or model from the loop is more efficient.

    print("Retraining best TabNet configuration on the full training set (X_train_np)...")
    # Note: TabNet's fit can be sensitive. For production, you'd save and load the best model from tuning.
    # Here we retrain for a representative number of epochs.
    retrain_epochs = max_epochs_per_trial // 2 # Example: half of tuning epochs
    final_tabnet_model.fit(
        X_train=X_train_np, y_train=y_train_np,
        # No eval_set here, training on all provided training data
        max_epochs=retrain_epochs, # Could be determined by best_model.best_epoch from tuning if available
        patience=0, # No early stopping for this final fit, train for fixed epochs
        batch_size=batch_size_tabnet,
        virtual_batch_size=virtual_batch_size_tabnet,
        num_workers=0,
        drop_last=False
    )
    final_tabnet_model.network.eval() # Set to eval mode

    # --- Evaluation on Test Set ---
    print("\n--- Evaluating Final TabNet on Test Set ---")
    y_pred_tabnet_test = final_tabnet_model.predict(X_test_np)

    exp_end_time = time.time()
    print(f"TabNet Experiment took {(exp_end_time - exp_start_time)/60:.2f} minutes.")
    print("---" * 20)
    return final_tabnet_model, y_test_np, y_pred_tabnet_test, best_tabnet_config