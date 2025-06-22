# src/modeling/deepfm_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, backend as K, Model, optimizers
import keras_tuner as kt # Keras Tuner
from sklearn.utils.class_weight import compute_sample_weight # For class weighting
import time

# SEED should ideally be passed or set globally by main.py
# For this example, let's assume it's passed as an argument where needed.
DEVICE = "CPU" # TensorFlow will use GPU if available by default

def set_seeds(seed_value):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    # If using Python's random module elsewhere:
    # random.seed(seed_value)

# --- DeepFM Model Builder (adapted from your Keras Tuner build_tuner_model) ---
def build_deepfm_keras_model(hp, input_dim, num_target_classes):
    """Builds the DeepFM model structure for Keras Tuner or final training."""
    
    # Hyperparameters from Keras Tuner or fixed best values
    embed_k   = hp.get("embed_k") # hp.Choice or fixed value
    drop_rate = hp.get("drop_rate")
    d1        = hp.get("d1")
    d2        = hp.get("d2")
    l2_fm     = hp.get("l2_fm")
    lr        = hp.get("lr")

    inp = layers.Input(shape=(input_dim,), name="input_features")

    # --- Wide Component (Direct connections for features) ---
    # For LabelEncoded features, this direct dense layer might not be as meaningful
    # as when using OHE features. Often omitted if embeddings are used for all.
    # However, your script had it. If features are just integers, its effect is limited.
    # wide_logit = layers.Dense(num_target_classes, name="wide_logit")(inp)

    # --- Embedding Layer for all input features ---
    # This is how DeepFM typically handles categorical features when they are integer encoded.
    # Each feature gets its own embedding vector of size 'embed_k'.
    # We need the cardinality of each feature to define the input_dim for each embedding.
    # For now, assuming 'inp' is a flat vector of all features, and we create a shared embedding
    # or sum individual embeddings. A more typical DeepFM has separate Embedding layers per field.
    # Given your script's `layers.Dense(embed_k, use_bias=False)(inp)`, it seems like a
    # dense projection to an embedding space rather than distinct per-feature embeddings.
    # This is more like a pre-embedding dense layer.
    
    # Let's follow your script structure which seems to project all input features
    # into a shared embedding space first.
    shared_embedding_layer = layers.Dense(embed_k * input_dim, use_bias=False, name="shared_projection_to_embedding_space_elements") (inp)
    reshaped_embeddings = layers.Reshape((input_dim, embed_k), name="reshaped_to_feature_embeddings")(shared_embedding_layer)


    # --- FM Component (Factorization Machine) ---
    # Sum of squares - Square of sum
    summed_feature_embeddings = layers.Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name="sum_feature_embeddings")(reshaped_embeddings)
    squared_sum_feature_embeddings = layers.Lambda(lambda x: K.square(K.sum(x, axis=1, keepdims=True)), name="sq_sum_feat_emb")(reshaped_embeddings) # This should be K.square(summed_feature_embeddings)
    
    # Corrected FM logic:
    sum_of_embeddings = layers.Lambda(lambda x: K.sum(x, axis=1), name="sum_of_embeddings")(reshaped_embeddings) # (None, embed_k)
    square_of_sum = layers.Lambda(lambda x: K.square(x), name="square_of_sum")(sum_of_embeddings) # (None, embed_k)

    sum_of_squares = layers.Lambda(lambda x: K.sum(K.square(x), axis=1), name="sum_of_squares")(reshaped_embeddings) # (None, embed_k)
    
    fm_interaction_term = layers.Lambda(lambda x: 0.5 * K.sum(x[0] - x[1], axis=1, keepdims=True), name="fm_interaction")([square_of_sum, sum_of_squares])
    # The FM term should contribute to the logits for each class.
    # Your script had a Dense layer after fm_raw.
    fm_logit = layers.Dense(num_target_classes, kernel_regularizer=regularizers.l2(l2_fm), name="fm_logit")(fm_interaction_term)
    
    # For wide part, if not using direct dense from input, sum of first-order embeddings
    first_order_embeddings = layers.Dense(num_target_classes, name="first_order_logits_from_embeddings")(layers.Flatten(name="flatten_for_1st_order")(reshaped_embeddings))


    # --- Deep Component ---
    # The input to the deep component is typically the concatenated embeddings
    flattened_embeddings = layers.Flatten(name="flattened_embeddings_for_deep")(reshaped_embeddings)
    deep = layers.Dropout(drop_rate, name="deep_dropout_1")(flattened_embeddings)
    deep = layers.Dense(d1, activation="relu", name="deep_dense_1")(deep)
    deep = layers.Dropout(drop_rate, name="deep_dropout_2")(deep)
    deep = layers.Dense(d2, activation="relu", name="deep_dense_2")(deep)
    deep_logit = layers.Dense(num_target_classes, name="deep_logit")(deep)

    # --- Combine Components ---
    # Original script uses Add: wide_logit, fm_logit, afm_logit, deep_logit
    # We'll use first_order_embeddings (as wide), fm_logit, deep_logit. AFM part is complex and can be added later.
    # If your original `inp` to `wide_logit` was OHE, it's different.
    # Assuming label encoded inputs, `first_order_embeddings` is more appropriate for the "wide" part.
    
    # Simpler combination without AFM for now, closer to standard DeepFM
    # The "wide" part in DeepFM often refers to raw features OR sum of 1st order embeddings.
    # Let's make 'wide_logit' be the first_order_embeddings as an approximation of the wide part
    # when input features are label encoded and then embedded.
    wide_logit_from_embeddings = layers.Dense(num_target_classes, name="wide_logit_final")(flattened_embeddings) # Or sum of 1st order term from embeddings

    fused_logits = layers.Add(name="add_logits")([wide_logit_from_embeddings, fm_logit, deep_logit])
    output = layers.Activation("softmax", name="output_softmax")(fused_logits)

    model = Model(inputs=inp, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    return model


def tune_and_train_deepfm(
    X_train_np, y_train_np, # LabelEncoded NumPy arrays
    X_val_np, y_val_np,
    X_test_np, y_test_np,
    input_dim_features, # Number of features
    num_target_classes,
    target_encoder_classes, # For classification report
    original_feature_names, # For potential feature importance interpretation
    seed=42,
    # Keras Tuner specific config
    tuner_project_name="deepfm_tuning",
    max_tuner_epochs=30, # Max epochs for each Keras Tuner trial run
    tuner_factor=3,
    tuner_patience=4, # Early stopping patience for Keras Tuner trials
    # Final model training config
    final_model_epochs=50, # Epochs to train the best model
    final_model_patience=8, # Early stopping patience for final model
    batch_size_deepfm=128
    ):
    """
    Tunes DeepFM using Keras Tuner Hyperband, then trains and evaluates the best model.
    Returns the best trained Keras model and its predictions on the test set.
    """
    print("\n" + "---" * 10 + " DeepFM Experiment " + "---" * 10)
    exp_start_time = time.time()
    set_seeds(seed)

    # --- Keras Tuner Hyperband Search (adapted from your script) ---
    print("\n--- Starting Keras Tuner (Hyperband) for DeepFM ---")
    tuner_start_time = time.time()

    # Pass additional fixed args to build_deepfm_keras_model
    # Keras Tuner's HyperModel class is more explicit for this, but a function works
    # by having hp.get() access fixed values if not in hp.Choice/hp.Float etc.
    # Or, define the model builder as a class. For now, simple function.
    # We need a way to pass input_dim and num_target_classes to build_deepfm_keras_model
    # when Keras Tuner calls it. We can use a wrapper or a HyperModel class.

    # Using a HyperModel class for cleaner parameter passing
    class DeepFMHyperModel(kt.HyperModel):
        def __init__(self, inp_dim, n_classes):
            self.inp_dim = inp_dim
            self.n_classes = n_classes
        def build(self, hp):
            return build_deepfm_keras_model(hp, self.inp_dim, self.n_classes)

    tuner = kt.Hyperband(
        DeepFMHyperModel(input_dim_features, num_target_classes),
        objective="val_sparse_categorical_accuracy",
        max_epochs=max_tuner_epochs,
        factor=tuner_factor,
        directory=f"keras_tuner_dir_{int(time.time())}", # Unique dir for tuner
        project_name=tuner_project_name,
        overwrite=True,
        seed=seed
    )

    tuner.search(
        X_train_np.astype(np.float32), y_train_np, # Ensure float32 for Keras
        validation_data=(X_val_np.astype(np.float32), y_val_np),
        epochs=max_tuner_epochs, # Max epochs for each configuration in Hyperband
        batch_size=batch_size_deepfm,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy", patience=tuner_patience
        )]
    )
    print(f"Keras Tuner search took {(time.time() - tuner_start_time)/60:.2f} minutes.")

    best_hyperparams_kt = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n✅ Best DeepFM Hyperparameters found by Keras Tuner:", best_hyperparams_kt.values)

    # --- Train Final Model with Best Hyperparameters ---
    print("\n--- Training Final DeepFM model with best parameters ---")
    # Build the model with the best hyperparameters found.
    final_deepfm_model = tuner.hypermodel.build(best_hyperparams_kt) # Or build_deepfm_keras_model(best_hyperparams_kt, ...)

    # For sample weights if using class_weight: 'balanced' like logic for Keras
    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train_np)

    final_model_train_start_time = time.time()
    final_deepfm_model.fit(
        X_train_np.astype(np.float32), y_train_np,
        validation_data=(X_val_np.astype(np.float32), y_val_np),
        epochs=final_model_epochs,
        batch_size=batch_size_deepfm,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy", patience=final_model_patience, restore_best_weights=True
        )],
        sample_weight=sample_weights_train, # Apply class balancing logic
        verbose=2 # Show epoch progress for final training
    )
    print(f"Final DeepFM training took {(time.time() - final_model_train_start_time):.2f} seconds.")

    # --- Evaluation on Test Set ---
    print("\n--- Evaluating Final DeepFM on Test Set ---")
    loss_test, acc_test = final_deepfm_model.evaluate(X_test_np.astype(np.float32), y_test_np, verbose=0)
    print(f"Test Accuracy (from model.evaluate): {acc_test:.4f}")

    y_pred_probs_deepfm_test = final_deepfm_model.predict(X_test_np.astype(np.float32), verbose=0)
    y_pred_deepfm_test = np.argmax(y_pred_probs_deepfm_test, axis=1)

    exp_end_time = time.time()
    print(f"DeepFM Experiment took {(exp_end_time - exp_start_time)/60:.2f} minutes.")
    print("---" * 20)
    return final_deepfm_model, y_test_np, y_pred_deepfm_test, best_hyperparams_kt.values