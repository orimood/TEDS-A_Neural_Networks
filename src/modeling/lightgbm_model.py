# src/modeling/lightgbm_model.py

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np # Not strictly needed here but good practice
import time

def get_lightgbm_default_params(num_target_classes, seed=42):
    """
    Returns a dictionary of default LightGBM parameters for a baseline run.
    """
    return {
        'objective': 'multiclass',
        'num_class': num_target_classes,
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1, # No limit
        'n_estimators': 1000,  # High number, actual trees determined by early stopping
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'min_child_samples': 20,
        'random_state': seed,
        'n_jobs': -1,
        'verbose': -1 # Suppress LightGBM's own training verbosity during fit
    }

def train_lightgbm_with_early_stopping(
    X_train_np, y_train_np, # Encoded NumPy arrays
    X_val_np, y_val_np,     # Encoded NumPy arrays for validation
    categorical_feature_indices_list,
    lgbm_params_dict,       # Full dictionary of parameters including n_estimators
    early_stopping_rounds=50
    ):
    """
    Trains a LightGBM model with specified parameters and early stopping.
    Returns the trained LightGBM model.
    """
    print(f"\n--- Training LightGBM with Early Stopping ---")
    print(f"Using parameters: {lgbm_params_dict}")
    
    model = lgb.LGBMClassifier(**lgbm_params_dict)
    
    fit_start_time = time.time()
    model.fit(
        X_train_np, y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_metric='multi_logloss', # A common metric for multi-class LGBM
        categorical_feature=categorical_feature_indices_list,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=1) # verbose=1 to see ES action
        ]
    )
    fit_end_time = time.time()
    print(f"LightGBM training with early stopping took {fit_end_time - fit_start_time:.2f} seconds.")
    
    # best_iteration_ can be None if early stopping wasn't triggered (trained for full n_estimators)
    if model.best_iteration_ is not None:
        print(f"Best iteration found by early stopping: {model.best_iteration_}")
        # Model is already trained with the best number of iterations due to how LGBM's fit works with early stopping
    else:
        print(f"Early stopping not triggered. Model trained for full {lgbm_params_dict.get('n_estimators', 'N/A')} estimators.")
        
    return model

def tune_lightgbm_with_gridsearch(
    X_train_np, y_train_np, # Encoded NumPy arrays
    categorical_feature_indices_list,
    num_target_classes,
    cv_folds=3,
    seed=42,
    grid_search_verbose=2
    ):
    """
    Tunes LightGBM using GridSearchCV.
    Returns the best estimator found by GridSearchCV.
    """
    print("\n--- Starting GridSearchCV for LightGBM ---")
    
    base_estimator = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_target_classes,
        random_state=seed,
        n_jobs=-1,
        # Note: `boosting_type` can also be a parameter in the grid if you want to try 'dart' or 'goss'
    )

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [200, 300, 400], # Upper limit, early stopping can be used if an eval_set is passed via fit_params
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 40, 50],
        'max_depth': [-1, 10, 15],
        'colsample_bytree': [0.7, 0.8],
        'subsample': [0.7, 0.8, 0.9], # Added subsample
        'min_child_samples': [20, 30, 50], # Added 30
        'class_weight': [None, 'balanced'],
        # 'reg_alpha': [0, 0.01, 0.1], # L1 regularization
        # 'reg_lambda': [0, 0.01, 0.1] # L2 regularization
    }

    # Fit parameters to pass to the estimator's fit method during CV
    # This is crucial for passing categorical_feature correctly
    # Early stopping can also be passed via fit_params if you have a consistent validation set for CV.
    # However, GridSearchCV's standard CV doesn't use a fixed eval_set for early stopping for all folds easily.
    # So, n_estimators in param_grid effectively controls tree count here.
    fit_params_for_gs = {'categorical_feature': categorical_feature_indices_list}

    f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)

    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring=f1_macro_scorer,
        cv=cv_folds,
        verbose=grid_search_verbose,
        n_jobs=-1 # Parallelize CV folds
    )

    gs_start_time = time.time()
    grid_search.fit(X_train_np, y_train_np, **fit_params_for_gs)
    gs_end_time = time.time()
    
    print(f"GridSearchCV for LightGBM took {(gs_end_time - gs_start_time)/60:.2f} minutes.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validated Macro F1 score: {grid_search.best_score_:.4f}")
    
    # GridSearchCV automatically refits the best estimator on the whole training data
    return grid_search.best_estimator_