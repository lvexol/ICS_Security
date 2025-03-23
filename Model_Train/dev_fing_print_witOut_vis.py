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

# Add debug print function
def debug_print(message):
    print(f"DEBUG: {message}")
    sys.stdout.flush()  # Ensure output is immediately displayed

# 1. Data Loading and Initial Exploration
def load_data(filepath_or_dir, is_directory=False):
    """
    Load dataset(s) from CSV file(s)
    
    Parameters:
    filepath_or_dir - Path to a CSV file or directory containing CSV files
    is_directory - Boolean indicating if the path is a directory
    
    Returns:
    df - DataFrame containing the loaded data
    """
    debug_print(f"Attempting to load data from: {filepath_or_dir}")
    
    if is_directory:
        # Check if directory exists
        if not os.path.exists(filepath_or_dir) or not os.path.isdir(filepath_or_dir):
            debug_print(f"ERROR: Directory not found: {filepath_or_dir}")
            debug_print(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Directory not found: {filepath_or_dir}")
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(filepath_or_dir) if f.endswith('.csv')]
        if not csv_files:
            debug_print(f"ERROR: No CSV files found in directory: {filepath_or_dir}")
            raise FileNotFoundError(f"No CSV files found in directory: {filepath_or_dir}")
        
        debug_print(f"Found {len(csv_files)} CSV files in directory")
        
        # Load and concatenate all CSV files
        dfs = []
        for file in csv_files:
            file_path = os.path.join(filepath_or_dir, file)
            try:
                debug_print(f"Loading file: {file}")
                df = pd.read_csv(file_path)
                debug_print(f"Loaded {file} with shape: {df.shape}")
                dfs.append(df)
            except Exception as e:
                debug_print(f"ERROR loading file {file}: {e}")
                # Continue with other files even if one fails
        
        if not dfs:
            debug_print("ERROR: Failed to load any CSV files")
            raise ValueError("Failed to load any CSV files")
        
        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)
        debug_print(f"Combined dataset shape: {df.shape}")
    else:
        # Single file mode
        # Check if file exists
        if not os.path.exists(filepath_or_dir):
            debug_print(f"ERROR: Dataset file not found: {filepath_or_dir}")
            debug_print(f"Current working directory: {os.getcwd()}")
            debug_print(f"Files in current directory: {os.listdir()}")
            raise FileNotFoundError(f"Dataset file not found: {filepath_or_dir}")
            
        try:
            df = pd.read_csv(filepath_or_dir)
            debug_print(f"Successfully loaded dataset with shape: {df.shape}")
        except Exception as e:
            debug_print(f"ERROR loading data: {e}")
            raise
    
    # Basic data validation
    debug_print(f"Dataset columns: {df.columns.tolist()}")
    required_columns = ['src']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        debug_print(f"ERROR: Required columns missing: {missing_columns}")
        raise ValueError(f"Required columns missing from dataset: {missing_columns}")
    
    return df

# Function to split files for training and testing
def split_csv_files(data_dir, test_count=3):
    """
    Split CSV files in a directory into training and testing sets
    
    Parameters:
    data_dir - Directory containing CSV files
    test_count - Number of files to reserve for testing
    
    Returns:
    train_files - List of file paths for training
    test_files - List of file paths for testing
    """
    debug_print(f"Splitting CSV files in {data_dir} for training and testing")
    
    # Get all CSV files in the directory
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if len(csv_files) < test_count + 1:
        debug_print(f"WARNING: Not enough CSV files for desired split. Found {len(csv_files)}, need at least {test_count + 1}")
        test_count = max(1, len(csv_files) - 1)  # Ensure at least 1 file for training
    
    # Shuffle files to ensure random selection
    np.random.shuffle(csv_files)
    
    # Split into training and testing sets
    train_files = csv_files[:-test_count]
    test_files = csv_files[-test_count:]
    
    debug_print(f"Selected {len(train_files)} files for training and {len(test_files)} files for testing")
    debug_print(f"Training files: {[os.path.basename(f) for f in train_files]}")
    debug_print(f"Testing files: {[os.path.basename(f) for f in test_files]}")
    
    return train_files, test_files

# Function to load multiple CSV files
def load_multiple_csv_files(file_paths):
    """
    Load and combine multiple CSV files
    
    Parameters:
    file_paths - List of paths to CSV files
    
    Returns:
    df - Combined DataFrame
    """
    debug_print(f"Loading {len(file_paths)} CSV files")
    
    dfs = []
    for file_path in file_paths:
        try:
            debug_print(f"Loading file: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            debug_print(f"Loaded with shape: {df.shape}")
            dfs.append(df)
        except Exception as e:
            debug_print(f"ERROR loading file {file_path}: {e}")
            # Continue with other files even if one fails
    
    if not dfs:
        debug_print("ERROR: Failed to load any CSV files")
        raise ValueError("Failed to load any CSV files")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    debug_print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

# 2. Data Preprocessing
def preprocess_data(df, encoders=None, scaler=None, is_training=True):
    """
    Preprocess the data for model training or inference
    """
    debug_print(f"Preprocessing data with shape {df.shape}")
    
    # Initialize encoders and scaler if in training mode
    if is_training:
        debug_print("Initializing new encoders and scaler")
        encoders = {}
        scaler = StandardScaler()
    else:
        debug_print("Using provided encoders and scaler")
        if encoders is None or scaler is None:
            raise ValueError("Encoders and scaler must be provided for inference")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # 1. Process timestamp if it exists
    if 'timestamp' in df.columns:
        debug_print("Processing timestamp column")
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Extract hour and minute and create cyclical features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Create cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        
        debug_print("Created cyclical time features")
    
    # 2. Process categorical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in ['timestamp', 'src', 'SCADA_Tag']]
    debug_print(f"Found categorical columns: {categorical_cols}")
    
    for col in categorical_cols:
        debug_print(f"Processing categorical column: {col}")
        if is_training:
            # Training phase: fit new encoder
            encoder = LabelEncoder()
            df[f'{col}_encoded'] = encoder.fit_transform(df[col].fillna('unknown').astype(str))
            encoders[col] = encoder
            debug_print(f"Fitted new encoder for {col} with {len(encoder.classes_)} classes")
        else:
            # Inference phase: use existing encoder
            if col in encoders:
                encoder = encoders[col]
                # Handle unseen categories
                df[col] = df[col].fillna('unknown').astype(str)
                df[f'{col}_encoded'] = df[col].map(
                    lambda x: np.argmax(np.array(encoder.classes_) == x) if x in encoder.classes_ else len(encoder.classes_)
                )
                debug_print(f"Applied existing encoder for {col}")
            else:
                debug_print(f"WARNING: No encoder found for {col}, skipping")
                continue
    
    # 3. Process numeric columns
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
    
    # Create SCADA_Tag label - this is what we want to predict
    if 'SCADA_Tag' in df.columns:
        debug_print("Processing SCADA_Tag labels")
        if is_training:
            # Create new LabelEncoder specifically for SCADA_Tag labels
            tag_encoder = LabelEncoder()
            df['tag_label'] = tag_encoder.fit_transform(df['SCADA_Tag'].fillna('unknown').astype(str))
            encoders['SCADA_Tag'] = tag_encoder
            
            # Store the number of unique tags
            num_classes = len(tag_encoder.classes_)
            debug_print(f"Found {num_classes} unique SCADA tags for classification")
            debug_print(f"SCADA tag classes: {tag_encoder.classes_}")
        else:
            # For inference/testing, use the existing encoder
            if 'SCADA_Tag' in encoders:
                tag_encoder = encoders['SCADA_Tag']
                df['SCADA_Tag'] = df['SCADA_Tag'].fillna('unknown').astype(str)
                
                # Map SCADA tags to labels, handling unseen values
                df['tag_label'] = df['SCADA_Tag'].map(
                    lambda x: np.argmax(np.array(tag_encoder.classes_) == x) if x in tag_encoder.classes_ else len(tag_encoder.classes_)
                )
                debug_print(f"Applied existing SCADA_Tag encoder with {len(tag_encoder.classes_)} classes")
                num_classes = len(tag_encoder.classes_)
            else:
                debug_print("WARNING: No SCADA_Tag encoder found, using placeholder")
                df['tag_label'] = 0
                num_classes = 1
    else:
        # If SCADA_Tag is not available, use a placeholder
        debug_print("WARNING: 'SCADA_Tag' column not found, using placeholder tag label")
        df['tag_label'] = 0
        num_classes = 1
    
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
    labels = df['tag_label'].values
    
    debug_print(f"Preprocessed features shape: {features.shape}")
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
def train_model_with_cv(X, y, num_classes, n_splits=2, window_size=5, hyperparams=None):
    """
    Train the transformer model with cross-validation
    """
    debug_print(f"Starting cross-validation training with {n_splits} folds")
    
    if hyperparams is None:
        hyperparams = {
            'num_heads': 2,  # Reduced from 4 to 2
            'dim_head': 32,  # Reduced from 64 to 32
            'num_blocks': 1,  # Reduced from 2 to 1
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 128,  # Increased from 64 to 128
            'max_epochs': 2  # Reduced from 5 to 2 epochs
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
        
        # Get unique classes in the test data
        unique_classes = np.unique(np.concatenate([y_test, y_pred_classes]))
        debug_print(f"Found {len(unique_classes)} unique classes in test data: {unique_classes}")
        
        # If class_names is provided, filter it to match the actual classes
        if class_names is not None:
            # Make sure we have enough class names
            if max(unique_classes) < len(class_names):
                # Filter class_names to only include classes present in the data
                filtered_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
                debug_print(f"Using {len(filtered_class_names)} filtered class names")
                print(classification_report(y_test, y_pred_classes, target_names=filtered_class_names, labels=unique_classes))
            else:
                debug_print(f"Warning: Some class indices exceed available class names. Using indices instead.")
                print(classification_report(y_test, y_pred_classes))
        else:
            print(classification_report(y_test, y_pred_classes))
        
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
    try:
        debug_print("Starting main function")
        
        # Set parameters
        data_dir = "/home/vexo/Project/rewrite/ICS_Security/Dataset"  # Directory containing CSV files
        test_file_count = 1  # Just 1 test file
        window_size = 5  # Reduced from 10 to 5
        stride = 5
        
        # Limit the number of training files to prevent memory issues
        max_train_files = 2  # Reduced to just 2 files
        
        # Print current working directory and check for dataset directory
        debug_print(f"Current working directory: {os.getcwd()}")
        debug_print(f"Looking for dataset directory at: {data_dir}")
        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            debug_print(f"ERROR: Dataset directory not found at {data_dir}")
            debug_print(f"Files in current directory: {os.listdir()}")
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        # 1. Split files for training and testing
        debug_print("Step 1: Splitting files for training and testing")
        all_train_files, test_files = split_csv_files(data_dir, test_count=test_file_count)
        
        # Limit the number of training files to prevent memory issues
        train_files = all_train_files[:max_train_files]
        debug_print(f"Using {len(train_files)} out of {len(all_train_files)} available training files to prevent memory issues")
        
        # 2. Load training data
        debug_print("Step 2: Loading training data")
        train_df = load_multiple_csv_files(train_files)
        
        # Optional: Sample the data to reduce memory usage
        sample_fraction = 0.05  # Reduced to just 5% of the data
        if train_df.shape[0] > 10000:  # If more than 10,000 rows
            debug_print(f"Sampling {sample_fraction*100}% of the data to reduce memory usage")
            train_df = train_df.sample(frac=sample_fraction, random_state=42)
            debug_print(f"Sampled dataset shape: {train_df.shape}")
        
        # 3. Preprocess the training data
        debug_print("Step 3: Preprocessing training data")
        features, labels, encoders, scaler, num_classes, feature_cols = preprocess_data(train_df, is_training=True)
        
        # 4. Create sequences for training
        debug_print("Step 4: Creating sequences for training data")
        X, y = create_sequences(features, labels, window_size=window_size, stride=stride)
        
        # 5. Split training data into train and validation sets
        debug_print("Step 5: Splitting training data into train and validation sets")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        debug_print(f"Train set shape: {X_train.shape}")
        debug_print(f"Validation set shape: {X_val.shape}")
        
        # 6. Train model with cross-validation
        debug_print("Step 6: Training model")
        best_model, val_metrics = train_model_with_cv(
            X_train, y_train, 
            num_classes=num_classes, 
            window_size=window_size
        )
        
        # 7. Load and process test files one by one
        debug_print("Step 7: Evaluating model on test files")
        
        all_test_metrics = []
        for test_file in test_files:
            file_name = os.path.basename(test_file)
            debug_print(f"\nEvaluating on test file: {file_name}")
            
            # Load test file
            test_df = pd.read_csv(test_file)
            debug_print(f"Test file shape: {test_df.shape}")
            
            # Preprocess test data
            test_features, test_labels, _, _, _, _ = preprocess_data(
                test_df, encoders=encoders, scaler=scaler, is_training=False
            )
            
            # Create sequences
            X_test, y_test = create_sequences(
                test_features, test_labels, window_size=window_size, stride=stride
            )
            
            debug_print(f"Test sequences shape: {X_test.shape}")
            
            # Evaluate model
            class_names = encoders['SCADA_Tag'].classes_ if 'SCADA_Tag' in encoders else None
            y_pred_classes, metrics = evaluate_model(best_model, X_test, y_test, class_names)
            
            # Store metrics for this test file
            file_metrics = {
                'file': file_name,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score']
            }
            all_test_metrics.append(file_metrics)
        
        # 8. Display overall test results
        debug_print("\nOverall test results:")
        for metrics in all_test_metrics:
            debug_print(f"File: {metrics['file']}, Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")
        
        # Calculate average metrics
        avg_accuracy = np.mean([m['accuracy'] for m in all_test_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in all_test_metrics])
        debug_print(f"\nAverage test accuracy: {avg_accuracy:.4f}")
        debug_print(f"Average test F1 score: {avg_f1:.4f}")
        
        # 9. Save model and artifacts
        debug_print("Step 9: Saving model and artifacts")
        # Create model directory
        model_dir = os.path.join(os.getcwd(), "model")
        debug_print(f"Creating model directory at: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        
        save_model_and_artifacts(
            best_model, encoders, scaler, 
            feature_cols, num_classes, window_size,
            base_path=model_dir
        )
        
        # Save test metrics
        metrics_path = os.path.join(model_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                'test_files': [os.path.basename(f) for f in test_files],
                'file_metrics': all_test_metrics,
                'average_accuracy': float(avg_accuracy),
                'average_f1_score': float(avg_f1)
            }, f, indent=2)
        
        debug_print(f"Model training complete with average test accuracy: {avg_accuracy:.4f}")
        debug_print(f"All artifacts saved to: {model_dir}")
        
    except Exception as e:
        debug_print(f"ERROR in main function: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    debug_print("Main function completed successfully")
    return True

if __name__ == "__main__":
    debug_print("Starting script")
    success = main()
    debug_print(f"Script completed with status: {'SUCCESS' if success else 'FAILURE'}")