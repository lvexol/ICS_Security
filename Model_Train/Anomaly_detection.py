# anomaly_detection.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import optuna

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=30, num_layers=5, dropout=0.17044, num_classes=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = self.relu(out[:, -1, :])
        
        # FC layer
        out = self.fc(out)
        return out

# Define function to balance dataset
def balance_dataset(X, y):
    """
    Balance the dataset by upsampling the minority class
    """
    X_normal = X[y == 0]
    y_normal = y[y == 0]
    X_attack = X[y == 1]
    y_attack = y[y == 1]
    
    # Upsample minority class
    if len(X_attack) < len(X_normal):
        X_attack_upsampled, y_attack_upsampled = resample(
            X_attack, y_attack,
            replace=True,
            n_samples=len(X_normal),
            random_state=42
        )
        X_balanced = np.vstack((X_normal, X_attack_upsampled))
        y_balanced = np.hstack((y_normal, y_attack_upsampled))
    else:
        X_balanced = np.vstack((X_normal, X_attack))
        y_balanced = np.hstack((y_normal, y_attack))
    
    return X_balanced, y_balanced

# Define preprocessing function
def preprocess_data(df, sequence_length=20):
    """
    Preprocess the data including:
    - Creating sequences for LSTM
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Ensure 'label' column exists
    if 'label' not in df.columns:
        raise ValueError("'label' column not found in the dataset")
    
    # Extract features and labels
    features = df.drop(columns=['label'])
    labels = df['label']
    
    # Create sequences
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features.iloc[i:i+sequence_length].values)
        y.append(labels.iloc[i+sequence_length])
    
    return np.array(X), np.array(y)

# Define training function
def train_model(model, train_loader, val_loader, epochs=59, lr=0.00676, device='cpu'):
    """
    Train the LSTM model with early stopping
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Early stopping parameters
    patience = 35
    best_val_acc = 0
    counter = 0
    best_model = None
    
    # For plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_val_loss = val_loss/len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model:
        model.load_state_dict(best_model)
    
    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model

# Define evaluation function
def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the model on the test set
    """
    model.eval()
    correct = 0
    total = 0
    
    # For confusion matrix
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Anomaly']))
    
    return accuracy, precision, recall, f1

# Define objective function for Optuna
def objective(trial, X_train, y_train, X_val, y_val, device):
    """
    Objective function for hyperparameter optimization
    """
    # Hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 15, 70)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    batch_size = trial.suggest_int('batch_size', 15, 400)
    dropout = trial.suggest_float('dropout', 0.1, 0.9)
    lr = trial.suggest_float('lr', 0.001, 0.01)
    
    # Create model with suggested hyperparameters
    input_size = X_train.shape[2]
    model = LSTMModel(input_size, hidden_size, num_layers, dropout)
    model = model.to(device)
    
    # Create data loaders with suggested batch size
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(10):  # Reduced epochs for optimization
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def process_csv_file(file_path, model=None, best_params=None, device='cpu', output_dir='output'):
    """
    Process a single CSV file, train or update the model, and return the updated model
    """
    print(f"\nProcessing file: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Display the first few rows of the dataset
    print("Dataset preview:")
    print(df.head())
    
    # Data exploration
    print("\nDataset shape:", df.shape)
    print("\nColumns in the dataset:", df.columns.tolist())
    
    # Check if 'label' column exists - try different case variations
    label_column = None
    for col in df.columns:
        if col.lower() == 'label':
            label_column = col
            break
    
    if label_column is None:
        # If no label column found, check if there's a column that might contain label information
        # For example, columns named 'class', 'target', 'anomaly', etc.
        potential_label_columns = ['class', 'target', 'anomaly', 'attack', 'normal', 'category']
        for col in df.columns:
            if any(label_name in col.lower() for label_name in potential_label_columns):
                print(f"Found potential label column: {col}")
                label_column = col
                break
    
    if label_column is None:
        # If still no label column, ask if the last column should be used as label
        last_col = df.columns[-1]
        print(f"No label column found. The last column is '{last_col}'.")
        print(f"Unique values in this column: {df[last_col].unique()}")
        print(f"Using the last column '{last_col}' as the label column.")
        
        # Rename the last column to 'label'
        df = df.rename(columns={last_col: 'label'})
        
        # Check if the values are numeric
        if not pd.api.types.is_numeric_dtype(df['label']):
            print("Converting label column to numeric...")
            # Try to convert to numeric, if not possible, use label encoding
            try:
                df['label'] = pd.to_numeric(df['label'])
            except:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df['label'] = le.fit_transform(df['label'])
                print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    else:
        # If a label column was found but with different case, rename it
        if label_column != 'label':
            df = df.rename(columns={label_column: 'label'})
    
    # Check if 'label' column exists now
    if 'label' not in df.columns:
        print("Error: Could not identify or create a 'label' column. Skipping this file.")
        dummy_metrics = {
            'file': file_path,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'status': 'skipped - no label column'
        }
        return model, best_params, dummy_metrics
    
    # Class distribution
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Ensure label is binary (0 or 1)
    unique_labels = df['label'].unique()
    if len(unique_labels) > 2:
        print(f"Warning: Found {len(unique_labels)} unique labels. Converting to binary classification.")
        # If more than 2 classes, convert to binary (normal vs anomaly)
        # Assuming 0 is normal and everything else is anomaly
        df['label'] = (df['label'] != 0).astype(int)
    elif len(unique_labels) == 2 and not all(label in [0, 1] for label in unique_labels):
        # If 2 classes but not 0 and 1, convert to 0 and 1
        min_label = df['label'].min()
        df['label'] = (df['label'] != min_label).astype(int)
        print(f"Converted labels to binary: {min_label} -> 0, {df['label'].max()} -> 1")
    
    # Preprocess data - just create sequences
    print("\nCreating sequences for LSTM...")
    sequence_length = 20
    try:
        X, y = preprocess_data(df, sequence_length=sequence_length)
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Class distribution in y: {np.bincount(y)}")
        
        # If dataset is still too large after preprocessing, sample it
        if len(X) > 100000:
            print(f"Preprocessed dataset is large ({len(X)} sequences). Sampling to reduce memory usage.")
            # Get indices for each class
            normal_indices = np.where(y == 0)[0]
            anomaly_indices = np.where(y == 1)[0]
            
            # Sample from each class
            sampled_normal = np.random.choice(normal_indices, min(50000, len(normal_indices)), replace=False)
            sampled_anomaly = np.random.choice(anomaly_indices, min(50000, len(anomaly_indices)), replace=False)
            
            # Combine indices and sample data
            sampled_indices = np.concatenate([sampled_normal, sampled_anomaly])
            X = X[sampled_indices]
            y = y[sampled_indices]
            
            print(f"Sampled dataset size: {len(X)} sequences")
            print(f"New class distribution in y: {np.bincount(y)}")
    except MemoryError:
        print("Memory error during preprocessing. The dataset is too large for available memory.")
        print("Try reducing the dataset size or processing on a machine with more memory.")
        dummy_metrics = {
            'file': file_path,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'status': 'skipped - memory error'
        }
        return model, best_params, dummy_metrics
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Balance the training data
    print("\nBalancing dataset...")
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    print(f"Balanced training set: {X_train_balanced.shape}, {y_train_balanced.shape}")
    print(f"Class distribution after balancing: {np.bincount(y_train_balanced)}")
    
    # If no model exists yet or we want to optimize for each file
    if model is None or best_params is None:
        # Hyperparameter optimization with Optuna
        print("\nStarting hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        
        # Create a partial function with fixed arguments
        objective_func = lambda trial: objective(trial, X_train, y_train, X_val, y_val, device)
        
        study.optimize(objective_func, n_trials=10)  # Reduced for demonstration, use 100 for full optimization
        
        best_params = study.best_params
        print(f"\nBest hyperparameters: {best_params}")
        
        # Create model with best hyperparameters
        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        )
        model = model.to(device)
    else:
        # If the input size is different, we need to recreate the model
        if X_train.shape[2] != model.lstm.input_size:
            print(f"\nInput size changed from {model.lstm.input_size} to {X_train.shape[2]}. Creating new model...")
            input_size = X_train.shape[2]
            model = LSTMModel(
                input_size=input_size,
                hidden_size=best_params['hidden_size'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout']
            )
            model = model.to(device)
    
    # Create data loaders
    batch_size = best_params['batch_size']
    train_dataset = TensorDataset(torch.FloatTensor(X_train_balanced), torch.LongTensor(y_train_balanced))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train model
    print("\nTraining model...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=59,  # As specified in the document
        lr=best_params['lr'],
        device=device
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
    
    # Save model for this file
    file_basename = os.path.basename(file_path).split('.')[0]
    model_path = os.path.join(output_dir, f"lstm_model_{file_basename}.pth")
    torch.save(model.state_dict(), model_path)

    # Save model in H5 format
    try:
        import h5py
        
        # Create H5 file
        with h5py.File(os.path.join(output_dir, f"lstm_model_{file_basename}.h5"), 'w') as h5f:
            # Save model architecture as attributes
            h5f.attrs['input_size'] = model.lstm.input_size
            h5f.attrs['hidden_size'] = model.hidden_size
            h5f.attrs['num_layers'] = model.num_layers
            h5f.attrs['dropout'] = model.lstm.dropout if hasattr(model.lstm, 'dropout') else 0.0
            
            # Create a group for weights
            weights_group = h5f.create_group('weights')
            
            # Save all model parameters
            for name, param in model.state_dict().items():
                weights_group.create_dataset(name, data=param.cpu().numpy())
        
        print(f"Model for {file_basename} saved in H5 format")
    except ImportError:
        print("Warning: h5py package not found. Could not save model in H5 format.")
        print("To save in H5 format, install h5py: pip install h5py")
    except Exception as e:
        print(f"Error saving model in H5 format: {str(e)}")
    
    # Save metrics
    metrics = {
        'file': file_path,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return model, best_params, metrics

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Directory containing CSV files
    data_dir = "/home/vexo/Project/rewrite/ICS_Security/Data"  # Change this to your actual directory path
    output_dir = "output"  # Directory to save outputs
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the directory
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Initialize model and best parameters
    model = None
    best_params = None
    all_metrics = []
    
    # Process each CSV file
    for file_path in csv_files:
        model, best_params, metrics = process_csv_file(
            file_path=file_path,
            model=model,
            best_params=best_params,
            device=device,
            output_dir=output_dir
        )
        all_metrics.append(metrics)
    
    # Save the final model
    if model is not None:
        print("\nSaving final model...")
        torch.save(model.state_dict(), os.path.join(output_dir, "final_lstm_model.pth"))
        
        # Save model in ONNX format for broader compatibility
        dummy_input = torch.randn(1, 20, model.lstm.input_size).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            os.path.join(output_dir, "final_lstm_model.onnx"),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Save model in TorchScript format (JIT)
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(os.path.join(output_dir, "final_lstm_model.pt"))
        
        # Save model in H5 format
        try:
            import h5py
            with h5py.File(os.path.join(output_dir, "final_lstm_model.h5"), 'w') as f:
                for name, param in model.named_parameters():
                    f.create_dataset(name, data=param.cpu().detach().numpy())
            print("Model saved in H5 format")
        except ImportError:
            print("h5py not installed, skipping H5 format save")
    
    # Save metrics summary
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    
    # Plot metrics summary
    if len(all_metrics) > 1:
        plt.figure(figsize=(12, 8))
        
        # Filter out None values for plotting
        valid_metrics = [m for m in all_metrics if m['accuracy'] is not None]
        
        if valid_metrics:
            plt.subplot(2, 2, 1)
            plt.bar(range(len(valid_metrics)), [m['accuracy'] for m in valid_metrics])
            plt.xlabel('File Index')
            plt.ylabel('Accuracy')
            plt.title('Accuracy by File')
            
            plt.subplot(2, 2, 2)
            plt.bar(range(len(valid_metrics)), [m['precision'] for m in valid_metrics])
            plt.xlabel('File Index')
            plt.ylabel('Precision')
            plt.title('Precision by File')
            
            plt.subplot(2, 2, 3)
            plt.bar(range(len(valid_metrics)), [m['recall'] for m in valid_metrics])
            plt.xlabel('File Index')
            plt.ylabel('Recall')
            plt.title('Recall by File')
            
            plt.subplot(2, 2, 4)
            plt.bar(range(len(valid_metrics)), [m['f1'] for m in valid_metrics])
            plt.xlabel('File Index')
            plt.ylabel('F1 Score')
            plt.title('F1 Score by File')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "metrics_summary.png"))
        else:
            print("No valid metrics to plot. All files were skipped.")
        
        plt.close()
    
    print("\nDone!")

if __name__ == "__main__":
    main()