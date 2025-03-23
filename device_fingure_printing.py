import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import math
import pickle
import os
import json
import sys
import argparse
import torch
import h5py

# Add debug print function
def debug_print(message):
    print(f"DEBUG: {message}")
    sys.stdout.flush()  # Ensure output is immediately displayed

# 1. Data Loading and Initial Exploration
def load_data(filepath):
    """Load the dataset from CSV file"""
    debug_print(f"Attempting to load data from: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        debug_print(f"ERROR: Dataset file not found: {filepath}")
        debug_print(f"Current working directory: {os.getcwd()}")
        debug_print(f"Files in current directory: {os.listdir()}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
    try:
        df = pd.read_csv(filepath)
        debug_print(f"Successfully loaded dataset with shape: {df.shape}")
        debug_print(f"Dataset columns: {df.columns.tolist()}")
        
        # Basic data validation
        required_columns = ['src']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            debug_print(f"ERROR: Required columns missing: {missing_columns}")
            raise ValueError(f"Required columns missing from dataset: {missing_columns}")
        
        return df
    except Exception as e:
        debug_print(f"ERROR loading data: {e}")
        raise

# 2. Data Preprocessing
def preprocess_data(df, encoders=None, scaler=None, is_training=True):
    """
    Preprocess the SCADA network logs data
    
    Parameters:
    df - DataFrame containing network logs
    encoders - Dict of Label Encoders (if None, new ones will be created)
    scaler - StandardScaler (if None, a new one will be created)
    is_training - Boolean to indicate if this is training data or inference data
    
    Returns:
    features - Processed feature array
    labels - Label array (or None if is_training=False)
    encoders - Dict of fitted Label Encoders
    scaler - Fitted StandardScaler
    num_classes - Number of unique device classes
    feature_cols - List of feature column names
    """
    debug_print("Starting data preprocessing")
    
    # Create copies of data to avoid modifying the original
    df = df.copy()
    
    # Initialize containers
    if encoders is None and is_training:
        encoders = {}
        debug_print("Created new encoders dictionary")
    
    if scaler is None and is_training:
        scaler = StandardScaler()
        debug_print("Created new StandardScaler")
    
    # Convert date and time to datetime objects if they exist
    if 'date' in df.columns and 'time' in df.columns:
        try:
            debug_print("Converting date and time to datetime")
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], 
                                           format='%d%b%Y %H:%M:%S', 
                                           errors='coerce')
            
            # Extract time features
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            df['second'] = df['datetime'].dt.second
            
            # Create sine/cosine encoding for time features to capture cyclical nature
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
            debug_print("Successfully created time features")
        except Exception as e:
            debug_print(f"Warning: Error processing datetime columns: {e}")
            # Create dummy time features if datetime processing fails
            for col in ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']:
                if col not in df:
                    df[col] = 0.0
    else:
        debug_print("No date/time columns found, creating dummy time features")
        # Create dummy time features if datetime columns don't exist
        for col in ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']:
            if col not in df:
                df[col] = 0.0
                
    # Identify categorical columns that exist in the dataset
    potential_categorical_cols = ['orig', 'i/f_name', 'i/f_dir', 'src', 'dst', 
                                 'proto', 'appi_name', 'SCADA_Tag']
    categorical_cols = [col for col in potential_categorical_cols if col in df.columns]
    debug_print(f"Found categorical columns: {categorical_cols}")
    
    # Process categorical variables
    for col in categorical_cols:
        debug_print(f"Processing categorical column: {col}")
        if is_training:
            # Training phase: create and fit a new encoder
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            debug_print(f"Created encoder for {col} with {len(le.classes_)} classes")
        else:
            # Inference phase: use existing encoder
            if col in encoders:
                # Handle unseen categories
                df[col] = df[col].astype(str)
                df[f'{col}_encoded'] = df[col].map(
                    lambda x: -1 if x not in encoders[col].classes_ else encoders[col].transform([x])[0]
                )
                # Replace unseen categories with most common value
                if (df[f'{col}_encoded'] == -1).any():
                    most_common = np.bincount(
                        [idx for idx in range(len(encoders[col].classes_))]).argmax()
                    df.loc[df[f'{col}_encoded'] == -1, f'{col}_encoded'] = most_common
                debug_print(f"Applied existing encoder for {col}")
            else:
                # Handle the case where encoder wasn't created during training
                debug_print(f"Warning: No encoder found for column {col}, using zeros")
                df[f'{col}_encoded'] = 0
    
    # Extract Modbus function code as numeric if it exists
    if 'Modbus_Function_Code' in df.columns:
        debug_print("Processing Modbus function code")
        df['Modbus_Function_Code'] = pd.to_numeric(df['Modbus_Function_Code'], errors='coerce')
        df['Modbus_Function_Code'].fillna(0, inplace=True)
    
    # Identify numeric columns that exist in the dataset
    potential_numeric_cols = ['s_port', 'Modbus_Transaction_ID', 'Modbus_Function_Code']
    numeric_cols = [col for col in potential_numeric_cols if col in df.columns]
    debug_print(f"Found numeric columns: {numeric_cols}")
    
    # Process numeric fields
    if numeric_cols:
        debug_print("Processing numeric columns")
        if is_training:
            # Training phase: fit scaler
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))
            debug_print("Fitted scaler on numeric columns")
        else:
            # Inference phase: use existing scaler
            for col in numeric_cols:
                if col not in df.columns:
                    df[col] = 0  # Add missing columns with default value
            df[numeric_cols] = scaler.transform(df[numeric_cols].fillna(0))
            debug_print("Applied existing scaler to numeric columns")
    
    # Create device fingerprint label - using source IP as device identifier
    if is_training:
        if 'src' in df.columns:
            debug_print("Creating device labels from source IPs")
            # Create new LabelEncoder specifically for device labels
            device_encoder = LabelEncoder()
            df['device_label'] = device_encoder.fit_transform(df['src'].astype(str))
            encoders['device'] = device_encoder
            
            # Store the number of unique devices
            num_classes = len(device_encoder.classes_)
            debug_print(f"Found {num_classes} unique devices for classification")
            debug_print(f"Device classes: {device_encoder.classes_}")
        else:
            # If source IP is not available, use a placeholder
            debug_print("WARNING: 'src' column not found, using placeholder device label")
            df['device_label'] = 0
            num_classes = 1
    else:
        # For inference, we don't need labels
        df['device_label'] = 0  # Placeholder
        num_classes = len(encoders.get('device', {}).classes_) if 'device' in encoders else 0
        debug_print(f"Using {num_classes} device classes for inference")
    
    # Select features for model
    feature_cols = [col for col in df.columns if col.endswith('_encoded') or col in 
                   ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'] or 
                   col in numeric_cols]
    
    # Make sure we have some features
    if not feature_cols:
        debug_print("ERROR: No valid features found in the dataset")
        raise ValueError("No valid features found in the dataset")
    
    debug_print(f"Selected {len(feature_cols)} features: {feature_cols}")
    
    # Extract features and labels
    features = df[feature_cols].values
    labels = df['device_label'].values if is_training else None
    
    debug_print(f"Preprocessed features shape: {features.shape}")
    if labels is not None:
        debug_print(f"Labels shape: {labels.shape}")
    
    return features, labels, encoders, scaler, num_classes, feature_cols

# 3. Sequence Creation
def create_sequences(features, labels=None, window_size=10, stride=1, pad_sequences=True):
    """
    Convert features and labels into sequences for the transformer model
    """
    debug_print(f"Creating sequences with window_size={window_size}, stride={stride}")
    
    X, y = [], []
    
    # Ensure we don't lose data at the end if pad_sequences is True
    if pad_sequences and len(features) % window_size != 0:
        pad_size = window_size - (len(features) % window_size)
        debug_print(f"Padding features with {pad_size} rows")
        padding = np.zeros((pad_size, features.shape[1]))
        features = np.vstack([features, padding])
        if labels is not None:
            # Use the last label for padding
            label_padding = np.full(pad_size, labels[-1])
            labels = np.concatenate([labels, label_padding])
    
    # Create sequences
    sequence_count = 0
    for i in range(0, len(features) - window_size + 1, stride):
        X.append(features[i:i+window_size])
        if labels is not None:
            # Use most frequent label in the window
            y.append(np.bincount(labels[i:i+window_size]).argmax())
        sequence_count += 1
        
        # Print progress every 10000 sequences
        if sequence_count % 10000 == 0:
            debug_print(f"Created {sequence_count} sequences so far")
    
    # Ensure we have some sequences
    if len(X) == 0:
        debug_print(f"ERROR: No sequences created. Features shape: {features.shape}, window size: {window_size}")
        raise ValueError(f"No sequences created. Features shape: {features.shape}, window size: {window_size}")
    
    X_array = np.array(X)
    y_array = np.array(y) if labels is not None else None
    
    debug_print(f"Created {len(X)} sequences with shape {X_array.shape}")
    if y_array is not None:
        debug_print(f"Labels shape: {y_array.shape}")
    
    return X_array, y_array

# 4. Transformer Model Helper Functions
def positional_encoding(position, d_model):
    """
    Create positional encoding for transformer model
    """
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder(inputs, heads, dim_head, dropout=0):
    """
    Create a transformer encoder block
    """
    # Multi-head attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=dim_head, num_heads=heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    
    # Feed forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=4 * dim_head * heads, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    
    return x + res

# 5. Main Model Building Function
def build_transformer_model(input_shape, num_classes, num_heads=4, dim_head=64, num_blocks=2, dropout=0.1):
    """
    Build a transformer-based model for device fingerprinting
    """
    debug_print(f"Building transformer model with input_shape={input_shape}, num_classes={num_classes}")
    
    inputs = keras.Input(shape=input_shape)
    
    # Add positional encoding
    pos_encoding = positional_encoding(input_shape[0], input_shape[1])
    x = inputs + pos_encoding
    
    # Transformer blocks
    for i in range(num_blocks):
        debug_print(f"Adding transformer block {i+1}/{num_blocks}")
        x = transformer_encoder(x, heads=num_heads, dim_head=dim_head, dropout=dropout)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    
    # Ensure outputs match the number of classes
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    debug_print(f"Model output shape: {outputs.shape}")
    
    # Compile model with appropriate loss and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model

# 6. Model Training with Cross-Validation
def train_model_with_cv(X, y, num_classes, n_splits=5, window_size=5, hyperparams=None):
    """
    Train the transformer model with cross-validation
    """
    debug_print(f"Starting cross-validation training with {n_splits} folds")
    
    if hyperparams is None:
        hyperparams = {
            'num_heads': 4,
            'dim_head': 64,
            'num_blocks': 2,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 64,
            'max_epochs': 50
        }
        debug_print(f"Using default hyperparameters: {hyperparams}")
    
    # Initialize K-fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Track metrics across folds
    val_metrics = []
    best_val_acc = 0
    best_model = None
    fold = 1
    
    # For each fold
    for train_idx, val_idx in kf.split(X):
        debug_print(f"\n--- Fold {fold}/{n_splits} ---")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        debug_print(f"Training data shape: {X_train.shape}")
        debug_print(f"Validation data shape: {X_val.shape}")
        
        # Verify label ranges
        debug_print(f"Training labels range: {np.min(y_train)} to {np.max(y_train)}")
        debug_print(f"Validation labels range: {np.min(y_val)} to {np.max(y_val)}")
        
        # Check if we have the expected number of classes
        observed_classes = len(np.unique(np.concatenate([y_train, y_val])))
        debug_print(f"Observed classes in data: {observed_classes}")
        
        # If there's a mismatch, adjust num_classes
        if observed_classes > num_classes:
            debug_print(f"Warning: Observed more classes ({observed_classes}) than expected ({num_classes})")
            debug_print("Adjusting num_classes to match observed classes")
            num_classes = observed_classes
        
        # Build and compile the model
        try:
            model = build_transformer_model(
                input_shape=X_train.shape[1:],
                num_classes=num_classes,
                num_heads=hyperparams['num_heads'],
                dim_head=hyperparams['dim_head'],
                num_blocks=hyperparams['num_blocks'],
                dropout=hyperparams['dropout']
            )
            debug_print("Model built successfully")
        except Exception as e:
            debug_print(f"ERROR building model: {e}")
            raise
        
        # Custom learning rate scheduler with more gradual decay
        def lr_scheduler(epoch, lr):
            if epoch < 5:
                return lr  # Keep initial lr for the first few epochs
            else:
                return lr * 0.95  # 5% decay per epoch after that
        
        callbacks = [
            # Learning rate scheduler
            keras.callbacks.LearningRateScheduler(lr_scheduler),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            
            # Model checkpoint to save best model
            keras.callbacks.ModelCheckpoint(
                f"best_model_fold_{fold}.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        try:
            debug_print(f"Starting training for fold {fold}")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=hyperparams['max_epochs'],
                batch_size=hyperparams['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            debug_print(f"Training completed for fold {fold}")
        except Exception as e:
            debug_print(f"ERROR during training: {e}")
            raise
        
        # Evaluate the model
        try:
            debug_print(f"Evaluating model for fold {fold}")
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            val_metrics.append({'fold': fold, 'val_loss': val_loss, 'val_acc': val_acc})
            
            debug_print(f"Fold {fold} validation accuracy: {val_acc:.4f}")
        except Exception as e:
            debug_print(f"ERROR during evaluation: {e}")
            raise
        
        # Save the best model across folds
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            debug_print(f"New best model found with accuracy: {best_val_acc:.4f}")
        
        fold += 1
    
    # Display cross-validation results
    val_acc_mean = np.mean([m['val_acc'] for m in val_metrics])
    val_acc_std = np.std([m['val_acc'] for m in val_metrics])
    debug_print(f"\nCross-validation results:")
    debug_print(f"Mean validation accuracy: {val_acc_mean:.4f} ± {val_acc_std:.4f}")
    
    return best_model, val_metrics

# 7. Model Evaluation
def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate the model performance
    """
    debug_print("Evaluating model on test data")
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    
    # Predict on test data
    try:
        debug_print(f"Making predictions on test data with shape {X_test.shape}")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        
        debug_print(f"Test accuracy: {accuracy:.4f}")
        debug_print(f"Test F1 score: {f1:.4f}")
        
        # Print classification report
        debug_print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=class_names))
        
        # Return predictions and metrics
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        return y_pred_classes, metrics
    except Exception as e:
        debug_print(f"ERROR during evaluation: {e}")
        raise

# 8. Save and Load Model
def save_model_and_artifacts(model, encoders, scaler, feature_cols, num_classes, window_size, base_path="model"):
    """
    Save the model and all artifacts needed for inference
    """
    debug_print(f"Saving model and artifacts to {base_path}")
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(base_path, exist_ok=True)
        debug_print(f"Created directory: {base_path}")
        
        # Print absolute path
        abs_path = os.path.abspath(base_path)
        debug_print(f"Absolute path: {abs_path}")
        
        # Save model
        model_path = os.path.join(base_path, "scada_device_fingerprinting.h5")
        debug_print(f"Saving model to: {model_path}")
        model.save(model_path)
        debug_print(f"Model saved successfully: {os.path.exists(model_path)}")
        
        # Save encoders
        encoders_path = os.path.join(base_path, "encoders.pkl")
        debug_print(f"Saving encoders to: {encoders_path}")
        with open(encoders_path, "wb") as f:
            pickle.dump(encoders, f)
        debug_print(f"Encoders saved successfully: {os.path.exists(encoders_path)}")
        
        # Save scaler
        scaler_path = os.path.join(base_path, "scaler.pkl")
        debug_print(f"Saving scaler to: {scaler_path}")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        debug_print(f"Scaler saved successfully: {os.path.exists(scaler_path)}")
        
        # Save metadata
        metadata = {
            'feature_columns': feature_cols,
            'num_classes': int(num_classes),
            'window_size': int(window_size),
            'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = os.path.join(base_path, "metadata.json")
        debug_print(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        debug_print(f"Metadata saved successfully: {os.path.exists(metadata_path)}")
        
        debug_print("All artifacts saved successfully!")
        
    except Exception as e:
        debug_print(f"ERROR saving artifacts: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_model_and_artifacts(base_path="model"):
    """
    Load the model and all artifacts needed for inference
    """
    debug_print(f"Loading model and artifacts from {base_path}")
    
    try:
        # Load model
        model_path = os.path.join(base_path, "scada_device_fingerprinting.h5")
        debug_print(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        debug_print("Model loaded successfully")
        
        # Load encoders
        encoders_path = os.path.join(base_path, "encoders.pkl")
        debug_print(f"Loading encoders from: {encoders_path}")
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)
        debug_print("Encoders loaded successfully")
        
        # Load scaler
        scaler_path = os.path.join(base_path, "scaler.pkl")
        debug_print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        debug_print("Scaler loaded successfully")
        
        # Load metadata
        metadata_path = os.path.join(base_path, "metadata.json")
        debug_print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        debug_print("Metadata loaded successfully")
        
        return model, encoders, scaler, metadata
    except Exception as e:
        debug_print(f"ERROR loading artifacts: {e}")
        import traceback
        traceback.print_exc()
        raise

# 9. Main Function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run device fingerprinting on a CSV file using a trained LSTM model')
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV file')
    parser.add_argument('--model', '-m', required=True, help='Path to H5 model file')
    parser.add_argument('--output', '-o', help='Path to output CSV file (optional)')
    parser.add_argument('--sequence_length', '-s', type=int, default=20, help='Sequence length (default: 20)')
    parser.add_argument('--device', '-d', default='cpu', choices=['cpu', 'cuda'], help='Device to run inference on (default: cpu)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return 1
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Generate default output path if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_basename = os.path.basename(args.input).split('.')[0]
        args.output = f"{input_basename}_fingerprinting_{timestamp}.csv"
    
    try:
        # Load and preprocess data
        sequences, original_df, source_ips = preprocess_csv(args.input, args.sequence_length)
        
        # Load model
        model, device_mapping, num_classes = load_model_from_h5(args.model)
        
        # Run inference
        predictions, probabilities = run_inference(model, sequences, num_classes, device)
        
        # Generate results
        results = generate_results(
            predictions, 
            probabilities, 
            original_df, 
            source_ips,
            args.sequence_length, 
            device_mapping, 
            args.output
        )
        
        print(f"\nDevice fingerprinting completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())