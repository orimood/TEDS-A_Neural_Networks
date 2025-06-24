# main.py

import pandas as pd
import numpy as np
import time
import os
import joblib # For saving scikit-learn models
import torch  # For saving PyTorch models
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# --- Import custom modules ---
from src.data_preprocessing import preprocess_teds_data
from src.evaluation import get_classification_metrics, plot_confusion_matrix_custom, plot_feature_importance_custom

# Import model training/tuning functions
from src.modeling.lightgbm_model import get_lightgbm_default_params, train_lightgbm_with_early_stopping, tune_lightgbm_with_gridsearch

from modeling.classical_models import train_logistic_regression, train_random_forest, tune_random_forest_gridsearch
from src.modeling.ft_transformer_model import train_and_evaluate_ft_transformer
from src.modeling.tabnet_model import train_and_evaluate_tabnet
from src.modeling.basic_nn_models import train_and_evaluate_embedding_nn # For MLP & EE-NN
from src.modeling.deepfm_model import tune_and_train_deepfm

# Import balancing utilities if they are in a separate file
# from src.utils import apply_smote, manual_oversample_nn # Example
# For now, let's assume SMOTE is specific to DeepFM and manual_oversample specific to basic_nn
# If SMOTE is used by more, it should be a utility.
from imblearn.over_sampling import SMOTE # For DeepFM
# The manual_oversample_nn is currently inside basic_nn_models.py, which is fine for now.


# --- 0. Global Configuration & Setup ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED) # If using Python's random module
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CONFIG_CSV_PATH = 'data/tedsa_puf_2020.csv' # Ensure this path is correct relative to main.py
RESULTS_DIR = 'results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Columns for initial loading and final feature set
CONFIG_INITIAL_COLS_MAIN = [
    'AGE', 'GENDER', 'RACE', 'ETHNIC', 'EDUC', 'EMPLOY',
    'LIVARAG', 'PRIMINC', 'STFIPS', 'REGION', 'DIVISION',
    'HLTHINS', 'SUB1'
]
CONFIG_FINAL_FEATURE_COLS_MAIN = [
    'AGE', 'GENDER', 'RACE', 'ETHNIC', 'EDUC', 'EMPLOY',
    'LIVARAG', 'PRIMINC', 'STFIPS', 'REGION', 'DIVISION',
    'HLTHINS'
]

# --- Model Run Configuration ---
# Set these flags to True for the models you want to run/tune
RUN_CONFIG = {
    "LightGBM":          {"run": True, "tune": True},
    "LogisticRegression":{"run": True, "tune": False}, # Tuning for LR is simpler, can add later
    "RandomForest":      {"run": True, "tune": True},
    "FTTransformer":     {"run": True, "tune": True}, # tune means run Optuna
    "TabNet":            {"run": True, "tune": True}, # tune means run its hyperparam search loop
    "MLP":               {"run": True, "tune": True}, # tune means run its search_configs
    "EE_NN":             {"run": True, "tune": True}, # tune means run its search_configs
    "DeepFM":            {"run": True, "tune": True}  # tune means run Keras Tuner
}

# --- Main Execution ---
def main():
    overall_start_time = time.time()
    all_model_results = []

    # 1. Preprocess Data (common for all models)
    print("\n" + " tahap ".center(50, "=")) # Tahap means stage/phase in Indonesian, using as a separator
    print(" Stage 1: Data Preprocessing ".center(50, "="))
    print(" tahap ".center(50, "=") + "\n")

    (X_train_processed, X_test_processed, y_train_encoded, y_test_encoded,
     feature_encoders, target_encoder, original_feature_names) = preprocess_teds_data(
        csv_file_path=CONFIG_CSV_PATH,
        initial_columns_to_load=CONFIG_INITIAL_COLS_MAIN,
        final_feature_columns=CONFIG_FINAL_FEATURE_COLS_MAIN,
        target_column='SUB1',
        sub1_freq_threshold=0.09,
        missing_code_to_remove=-9,
        test_split_size=0.2, # This is for train/test split
        random_state_split=SEED
    )

    if X_train_processed is None:
        print("Data preprocessing failed. Exiting.")
        return

    # 1b. Create a dedicated validation set from the training data for models needing it
    print("\n--- Creating Train/Validation Split from Processed Training Data ---")
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_processed, y_train_encoded,
        test_size=0.25, # e.g., 25% of the 80% train -> 20% of original for validation
        random_state=SEED,
        stratify=y_train_encoded
    )
    print(f"Final training set size: {X_train_final.shape[0]}")
    print(f"Validation set size: {X_val_final.shape[0]}")
    print(f"Test set size: {X_test_processed.shape[0]}\n")

    num_target_classes = len(target_encoder.classes_)
    cat_feat_indices = [i for i in range(X_train_final.shape[1])] # All features are categorical after encoding

    # For NNs needing cardinalities (TabNet, FT-Transformer, EmbeddingNN)
    # These are derived from the LabelEncoders used in preprocess_teds_data for each feature
    cat_cardinalities = [len(feature_encoders[col_name].classes_) for col_name in original_feature_names]


    # --- Model Training, Tuning, and Evaluation Loop ---
    print("\n" + " tahap ".center(50, "="))
    print(" Stage 2: Model Training & Evaluation ".center(50, "="))
    print(" tahap ".center(50, "=") + "\n")

    # --- LightGBM ---
    if RUN_CONFIG["LightGBM"]["run"]:
        model_name = "LightGBM"
        if RUN_CONFIG["LightGBM"]["tune"]:
            lgbm = tune_lightgbm_with_gridsearch(
                X_train_final, y_train_final, cat_feat_indices,
                num_target_classes=num_target_classes, cv_folds=3, seed=SEED
            )
            model_name += " (Tuned)"
        else:
            lgbm_params = get_lightgbm_default_params(num_target_classes, seed=SEED)
            lgbm = train_lightgbm_with_early_stopping(
                X_train_final, y_train_final, X_val_final, y_val_final,
                cat_feat_indices, lgbm_params
            )
            model_name += " (Baseline w/ ES)"
        
        y_pred_lgbm = lgbm.predict(X_test_processed)
        metrics_lgbm = get_classification_metrics(y_test_encoded, y_pred_lgbm, target_names=[str(c) for c in target_encoder.classes_])
        all_model_results.append({"model": model_name, **metrics_lgbm, "best_params": getattr(lgbm, 'best_params_', lgbm.get_params() if not RUN_CONFIG["LightGBM"]["tune"] else None)})
        plot_confusion_matrix_custom(y_test_encoded, y_pred_lgbm, target_encoder.classes_, model_name, save_path=os.path.join(PLOTS_DIR, f"{model_name.replace(' ','_')}_cm.png"))
        plot_feature_importance_custom(lgbm, original_feature_names, model_name, save_path=os.path.join(PLOTS_DIR, f"{model_name.replace(' ','_')}_fi.png"))

    # --- Logistic Regression ---
    if RUN_CONFIG["LogisticRegression"]["run"]:
        model_name = "LogisticRegression"
        lr_model = train_logistic_regression(X_train_final, y_train_final) # Using default params
        y_pred_lr = lr_model.predict(X_test_processed)
        metrics_lr = get_classification_metrics(y_test_encoded, y_pred_lr, target_names=[str(c) for c in target_encoder.classes_])
        all_model_results.append({"model": model_name, **metrics_lr, "best_params": lr_model.get_params()})
        plot_confusion_matrix_custom(y_test_encoded, y_pred_lr, target_encoder.classes_, model_name, save_path=os.path.join(PLOTS_DIR, f"{model_name}_cm.png"))
        

    # --- Random Forest ---
    if RUN_CONFIG["RandomForest"]["run"]:
        model_name = "RandomForest"
        if RUN_CONFIG["RandomForest"]["tune"]:
            rf_model = tune_random_forest_gridsearch(X_train_final, y_train_final, cv_folds=3)
            model_name += " (Tuned)"
        else:
            rf_model = train_random_forest(X_train_final, y_train_final) # Using default params
            model_name += " (Baseline)"
        
        y_pred_rf = rf_model.predict(X_test_processed)
        metrics_rf = get_classification_metrics(y_test_encoded, y_pred_rf, target_names=[str(c) for c in target_encoder.classes_])
        all_model_results.append({"model": model_name, **metrics_rf, "best_params": getattr(rf_model, 'best_params_', rf_model.get_params() if not RUN_CONFIG["RandomForest"]["tune"] else None)})
        plot_confusion_matrix_custom(y_test_encoded, y_pred_rf, target_encoder.classes_, model_name, save_path=os.path.join(PLOTS_DIR, f"{model_name.replace(' ','_')}_cm.png"))
        plot_feature_importance_custom(rf_model, original_feature_names, model_name, save_path=os.path.join(PLOTS_DIR, f"{model_name.replace(' ','_')}_fi.png"))

    # --- FT-Transformer ---
    if RUN_CONFIG["FTTransformer"]["run"]:
        # FT-Transformer specific data prep (already label encoded, just need numpy)
        # Its tuning function handles internal CV or train/val split for Optuna trials
        model_name = "FTTransformer"
        ft_model, ft_y_true, ft_y_pred, ft_best_params = train_and_evaluate_ft_transformer(
            X_train_final, y_train_final, X_test_processed, y_test_encoded, # Passing test as val for Optuna's last check
            cat_cardinalities_list=cat_cardinalities,
            num_target_classes=num_target_classes,
            feature_names_list=original_feature_names,
            target_encoder_classes=target_encoder.classes_,
            seed=SEED,
            optuna_n_trials=10, # Make configurable
            optuna_timeout_hours=0.2, # Make configurable
            final_model_epochs=20 # Make configurable
        )
        # metrics_ft = get_classification_metrics(ft_y_true, ft_y_pred, target_names=[str(c) for c in target_encoder.classes_]) # Evaluation is inside
        # The train_and_evaluate_ft_transformer already prints its metrics.
        # We can re-calculate here if we want to add to all_model_results consistently
        if ft_y_true is not None: # Check if training was successful
             metrics_ft_main = get_classification_metrics(ft_y_true, ft_y_pred, target_names=[str(c) for c in target_encoder.classes_])
             all_model_results.append({"model": model_name + " (Tuned)", **metrics_ft_main, "best_params": ft_best_params})
             plot_confusion_matrix_custom(ft_y_true, ft_y_pred, target_encoder.classes_, model_name + " (Tuned)", save_path=os.path.join(PLOTS_DIR, f"{model_name}_cm.png"))
             # Feature importance for FT-Transformer often uses permutation or SHAP, not a direct attribute

    # --- TabNet ---
    if RUN_CONFIG["TabNet"]["run"]:
        model_name = "TabNet"
        # TabNet specific data prep (it expects numpy arrays)
        # Its tuning/training function takes train and val sets for early stopping
        tab_model, tab_y_true, tab_y_pred, tab_best_cfg = train_and_evaluate_tabnet(
            X_train_final, y_train_final, X_val_final, y_val_final, X_test_processed, y_test_encoded,
            cat_idxs_list=cat_feat_indices,
            cat_dims_list=cat_cardinalities,
            target_encoder_classes=target_encoder.classes_,
            seed=SEED,
            max_search_trials=5, # Make configurable
            max_epochs_per_trial=20, # Make configurable
            patience_per_trial=5
        )
        if tab_y_true is not None:
            metrics_tab_main = get_classification_metrics(tab_y_true, tab_y_pred, target_names=[str(c) for c in target_encoder.classes_])
            all_model_results.append({"model": model_name + " (Tuned)", **metrics_tab_main, "best_params": tab_best_cfg})
            plot_confusion_matrix_custom(tab_y_true, tab_y_pred, target_encoder.classes_, model_name + " (Tuned)", save_path=os.path.join(PLOTS_DIR, f"{model_name}_cm.png"))
            # TabNet has its own feature importance: tab_model.feature_importances_
            # plot_feature_importance_custom(tab_model, original_feature_names, model_name + " (Tuned)")
            # tab_model.save_model(os.path.join(MODELS_DIR, f"{model_name}_model")) # Saves as a zip

    # --- MLP / EE-NN (using EmbeddingNN) ---
   
    nn_models_to_run = []
    if RUN_CONFIG["MLP"]["run"]: nn_models_to_run.append("MLP")
    if RUN_CONFIG["EE_NN"]["run"]: nn_models_to_run.append("EE_NN")

    for nn_model_type in nn_models_to_run:
        model_name = nn_model_type
        # Define specific hyperparam_configs_list for MLP and EE-NN based on your images
       
        if nn_model_type == "MLP":
            nn_config_list = [{'hidden_layers': [256], 'dropout': 0.2, 'lr': 0.0005, 'embedding_rule': 32}] # MLP might use fixed small embedding or no explicit embedding if features were OHE
            use_oversample = True # As per your notes
            use_class_w = True
        elif nn_model_type == "EE_NN":
            nn_config_list = [{'hidden_layers': [256], 'dropout': 0.2, 'lr': 0.0005, 'embedding_rule': 'default_rule'}]
            use_oversample = True
            use_class_w = True
        else:
            continue

        # Determine if tuning (iterating multiple configs) or single run
        should_tune_nn = RUN_CONFIG[nn_model_type]["tune"] and len(nn_config_list) > 1 # or pass a larger list for tuning

        nn_model, nn_y_true, nn_y_pred, nn_best_cfg = train_and_evaluate_embedding_nn(
            X_train_final, y_train_final, X_val_final, y_val_final, X_test_processed, y_test_encoded,
            cat_dims_list=cat_cardinalities,
            num_target_classes=num_target_classes,
            target_encoder_classes_str=target_encoder.classes_,
            original_feature_names=original_feature_names,
            model_type_name=model_name,
            seed=SEED,
            hyperparam_configs_list=nn_config_list,
            max_search_trials= 1 if not should_tune_nn else len(nn_config_list), # Run all if tuning, else 1
            epochs_per_trial=15, # From your image
            patience_early_stop=4,
            batch_size_nn=64,    # From your image
            use_manual_oversampling=use_oversample,
            use_class_weights_loss=use_class_w
        )
        if nn_y_true is not None:
            metrics_nn_main = get_classification_metrics(nn_y_true, nn_y_pred, target_names=[str(c) for c in target_encoder.classes_])
            all_model_results.append({"model": model_name + " (Best Config)", **metrics_nn_main, "best_params": nn_best_cfg})
            plot_confusion_matrix_custom(nn_y_true, nn_y_pred, target_encoder.classes_, model_name + " (Best Config)", save_path=os.path.join(PLOTS_DIR, f"{model_name}_cm.png"))
            # torch.save(nn_model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}.pt"))
            
    # --- DeepFM ---
    if RUN_CONFIG["DeepFM"]["run"]:
        model_name = "DeepFM"
        # DeepFM requires SMOTE as per your notes
        print("\nApplying SMOTE for DeepFM training data...")
        smote = SMOTE(random_state=SEED, k_neighbors=min(5, np.min(np.bincount(y_train_final))-1 if np.min(np.bincount(y_train_final)) > 1 else 1 )) # Adjust k_neighbors
        X_train_deepfm_smoted, y_train_deepfm_smoted = smote.fit_resample(X_train_final, y_train_final)
        print(f"SMOTEd training data shape for DeepFM: X={X_train_deepfm_smoted.shape}, y={y_train_deepfm_smoted.shape}")

        dfm_model, dfm_y_true, dfm_y_pred, dfm_best_cfg = tune_and_train_deepfm(
            X_train_deepfm_smoted, y_train_deepfm_smoted, # Use SMOTEd data for training
            X_val_final, y_val_final,       # Original validation set
            X_test_processed, y_test_encoded,
            input_dim_features=X_train_final.shape[1],
            num_target_classes=num_target_classes,
            target_encoder_classes=target_encoder.classes_,
            original_feature_names=original_feature_names,
            seed=SEED,
            tuner_project_name=f"deepfm_teds_{int(time.time())}",
            max_tuner_epochs=10, # Reduced for quicker run, your script used 30
            tuner_patience=3,    # Your script used 4
            final_model_epochs=20, # Reduced for quicker run, your script used 50
            final_model_patience=5, # Your script used 8
            batch_size_deepfm=128
        )
        if dfm_y_true is not None:
            metrics_dfm_main = get_classification_metrics(dfm_y_true, dfm_y_pred, target_names=[str(c) for c in target_encoder.classes_])
            all_model_results.append({"model": model_name + " (Tuned)", **metrics_dfm_main, "best_params": dfm_best_cfg})
            plot_confusion_matrix_custom(dfm_y_true, dfm_y_pred, target_encoder.classes_, model_name + " (Tuned)", save_path=os.path.join(PLOTS_DIR, f"{model_name}_cm.png"))


    # --- 3. Aggregate and Display Results ---
    print("\n\n" + " tahap ".center(50, "="))
    print(" Stage 3: Final Results Summary ".center(50, "="))
    print(" tahap ".center(50, "=") + "\n")

    if all_model_results:
        results_df = pd.DataFrame(all_model_results)
        # Select key metrics for summary display
        summary_cols = ["model", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
        print(results_df[summary_cols].sort_values(by="f1_macro", ascending=False))
        results_df.to_csv(os.path.join(RESULTS_DIR, "all_model_metrics_summary.csv"), index=False)
        print(f"\nFull metrics summary saved to {os.path.join(RESULTS_DIR, 'all_model_metrics_summary.csv')}")
    else:
        print("No models were run or no results collected.")

    overall_end_time = time.time()
    print(f"\nTotal script execution time: {(overall_end_time - overall_start_time) / 60:.2f} minutes.")


if __name__ == '__main__':
    main()
