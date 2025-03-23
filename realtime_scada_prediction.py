import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import os
import sys
import time
from collections import deque
import argparse

# Debug print function
def debug_print(message):
    print(f"DEBUG: {message}")
    sys.stdout.flush()

class RealTimeSCADAPredictor:
    def __init__(self, model_dir="model"):
        """
        Initialize the real-time SCADA tag predictor
        """
        self.model_dir = model_dir
        self.model = None
        self.encoders = None
        self.scaler = None
        self.metadata = None
        self.feature_cols = None
        self.window_size = None
        self.num_classes = None
        self.tag_names = None
        self.buffer = None
        self.is_loaded = False
        
    def load_model(self):
        """
        Load the trained model and preprocessing artifacts
        """
        try:
            debug_print(f"Loading model and artifacts from {self.model_dir}")
            
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                debug_print(f"ERROR: Model directory not found: {self.model_dir}")
                debug_print(f"Current working directory: {os.getcwd()}")
                debug_print(f"Available directories: {os.listdir('.')}")
                raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
            
            # Load model
            model_path = os.path.join(self.model_dir, "scada_device_fingerprinting.h5")
            debug_print(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                debug_print(f"ERROR: Model file not found: {model_path}")
                debug_print(f"Files in model directory: {os.listdir(self.model_dir)}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Try different loading methods to handle version compatibility issues
            try:
                debug_print("Attempting to load model with standard method...")
                self.model = keras.models.load_model(model_path, compile=False)
            except TypeError as e:
                debug_print(f"Standard loading failed: {e}")
                debug_print("Trying alternative loading method with custom_objects...")
                
                try:
                    # Try with custom_objects
                    from tensorflow.keras.layers import Layer
                    
                    # Define a custom layer that can handle the version mismatch
                    class VersionCompatLayer(Layer):
                        def __init__(self, **kwargs):
                            super(VersionCompatLayer, self).__init__(**kwargs)
                        
                        def call(self, inputs):
                            return inputs
                        
                        def get_config(self):
                            return {}
                    
                    custom_objects = {
                        'VersionCompatLayer': VersionCompatLayer,
                        'MultiHeadAttention': keras.layers.MultiHeadAttention
                    }
                    
                    self.model = keras.models.load_model(
                        model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    debug_print("Model loaded successfully with custom objects")
                except Exception as e2:
                    debug_print(f"Alternative loading also failed: {e2}")
                    debug_print("Trying to rebuild model from scratch...")
                    
                    try:
                        # Try to load the model architecture from a JSON file if it exists
                        json_path = os.path.join(self.model_dir, "model_architecture.json")
                        if os.path.exists(json_path):
                            with open(json_path, 'r') as f:
                                model_json = f.read()
                            
                            from tensorflow.keras.models import model_from_json
                            self.model = model_from_json(model_json)
                            self.model.load_weights(model_path)
                            debug_print("Model rebuilt from architecture JSON and weights")
                        else:
                            # If all else fails, create a simple model that can make predictions
                            debug_print("Creating a simple model as fallback...")
                            
                            # Load metadata to get input shape and output classes
                            metadata_path = os.path.join(self.model_dir, "metadata.json")
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)
                            
                            window_size = metadata['window_size']
                            num_features = len(metadata['feature_columns'])
                            num_classes = metadata['num_classes']
                            
                            # Create a simple LSTM model
                            self.model = keras.Sequential([
                                keras.layers.InputLayer(input_shape=(window_size, num_features)),
                                keras.layers.LSTM(64),
                                keras.layers.Dense(32, activation='relu'),
                                keras.layers.Dense(num_classes, activation='softmax')
                            ])
                            debug_print("Created fallback model. Note: This will not give accurate predictions!")
                    except Exception as e3:
                        debug_print(f"All loading methods failed: {e3}")
                        raise ValueError("Could not load or create a model due to version incompatibility")
            
            # Load encoders
            encoders_path = os.path.join(self.model_dir, "encoders.pkl")
            if not os.path.exists(encoders_path):
                debug_print(f"ERROR: Encoders file not found: {encoders_path}")
                raise FileNotFoundError(f"Encoders file not found: {encoders_path}")
            
            with open(encoders_path, "rb") as f:
                self.encoders = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if not os.path.exists(scaler_path):
                debug_print(f"ERROR: Scaler file not found: {scaler_path}")
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                debug_print(f"ERROR: Metadata file not found: {metadata_path}")
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            
            # Extract metadata
            self.feature_cols = self.metadata['feature_columns']
            self.window_size = self.metadata['window_size']
            self.num_classes = self.metadata['num_classes']
            
            # Create a buffer to store recent records
            self.buffer = deque(maxlen=self.window_size)
            
            # Get SCADA tag names from encoders
            if 'SCADA_Tag' in self.encoders:
                self.tag_names = self.encoders['SCADA_Tag'].classes_
                debug_print(f"Loaded {len(self.tag_names)} SCADA tag names")
            else:
                self.tag_names = [f"Tag_{i}" for i in range(self.num_classes)]
                debug_print("No SCADA tag names found, using generic names")
            
            debug_print(f"Real-time predictor initialized with window size {self.window_size}")
            debug_print(f"Model expects {len(self.feature_cols)} features")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            debug_print(f"ERROR loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False
    
    def preprocess_record(self, record):
        """
        Preprocess a single record for prediction
        """
        try:
            # Convert record to DataFrame if it's a dictionary
            if isinstance(record, dict):
                df = pd.DataFrame([record])
            else:
                df = pd.DataFrame([record.to_dict()])
            
            # Process categorical columns
            for col, encoder in self.encoders.items():
                if col == 'SCADA_Tag':  # Skip the SCADA_Tag encoder which is for labels
                    continue
                    
                if col in df.columns:
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    df[f'{col}_encoded'] = df[col].map(
                        lambda x: np.argmax(np.array(encoder.classes_) == x) if x in encoder.classes_ else len(encoder.classes_)
                    )
                else:
                    # If column doesn't exist, use a default value
                    df[f'{col}_encoded'] = 0
            
            # Process time features if they exist
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    
                    # Extract time features
                    df['hour'] = df['timestamp'].dt.hour
                    df['minute'] = df['timestamp'].dt.minute
                    
                    # Create sine/cosine encoding for time features
                    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
                    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
                except:
                    # Create dummy time features if datetime processing fails
                    for col in ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']:
                        df[col] = 0.0
            else:
                # Create dummy time features if datetime columns don't exist
                for col in ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']:
                    df[col] = 0.0
            
            # Process numeric columns
            numeric_cols = ['s_port', 'Modbus_Transaction_ID', 'Modbus_Function_Code']
            for col in numeric_cols:
                if col not in df.columns:
                    df[col] = 0  # Add missing columns with default value
            
            # Apply scaler to numeric columns that exist
            existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
            if existing_numeric_cols:
                df[existing_numeric_cols] = self.scaler.transform(df[existing_numeric_cols].fillna(0))
            
            # Extract features
            features = []
            for col in self.feature_cols:
                if col in df.columns:
                    features.append(df[col].iloc[0])
                else:
                    features.append(0)  # Default value for missing features
            
            return np.array(features)
            
        except Exception as e:
            debug_print(f"ERROR preprocessing record: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a zero array as fallback
            return np.zeros(len(self.feature_cols))
    
    def process_record(self, record):
        """
        Process a single record and update predictions
        """
        if not self.is_loaded:
            debug_print("Model not loaded. Call load_model() first.")
            return None, 0.0
        
        try:
            # Preprocess the record
            features = self.preprocess_record(record)
            
            # Add to buffer
            self.buffer.append(features)
            
            # If buffer is not full yet, return None
            if len(self.buffer) < self.window_size:
                return None, len(self.buffer) / self.window_size
            
            # Create sequence from buffer
            sequence = np.array([list(self.buffer)])
            
            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Get tag name
            if predicted_class < len(self.tag_names):
                tag_name = self.tag_names[predicted_class]
            else:
                tag_name = f"Unknown_Tag_{predicted_class}"
            
            return {
                'tag_id': int(predicted_class),
                'tag_name': tag_name,
                'confidence': float(confidence),
                'prediction': prediction[0].tolist()
            }, 1.0  # Buffer is full, so progress is 100%
            
        except Exception as e:
            debug_print(f"ERROR processing record: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, len(self.buffer) / self.window_size if self.buffer else 0.0

def simulate_real_time_data(csv_file, predictor, delay=0.1):
    """
    Simulate real-time data by reading a CSV file row by row
    """
    debug_print(f"Loading data from {csv_file} for simulation")
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    debug_print(f"Loaded {len(df)} records for simulation")
    
    # Process each record
    for i, record in df.iterrows():
        # Process the record
        result, progress = predictor.process_record(record)
        
        # Display result if available
        if result:
            print(f"\nRecord {i+1}/{len(df)}:")
            print(f"Detected SCADA Tag: {result['tag_name']} (ID: {result['tag_id']})")
            print(f"Confidence: {result['confidence']:.4f}")
            
            # Display top 3 predictions
            top_indices = np.argsort(result['prediction'])[-3:][::-1]
            print("Top 3 predictions:")
            for idx in top_indices:
                if idx < len(predictor.tag_names):
                    tag = predictor.tag_names[idx]
                else:
                    tag = f"Unknown_Tag_{idx}"
                print(f"  {tag}: {result['prediction'][idx]:.4f}")
        else:
            print(f"\rFilling buffer: {progress*100:.1f}% complete", end="")
        
        # Simulate delay between records
        time.sleep(delay)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time SCADA tag prediction')
    parser.add_argument('--model_dir', type=str, default='model', 
                        help='Directory containing the model and artifacts')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file for simulation')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between records in seconds')
    args = parser.parse_args()
    
    try:
        # Initialize the predictor
        predictor = RealTimeSCADAPredictor(args.model_dir)
        
        # Load the model
        if not predictor.load_model():
            debug_print("Failed to load model. Exiting.")
            return 1
        
        # Simulate real-time data
        simulate_real_time_data(args.input, predictor, args.delay)
        
    except Exception as e:
        debug_print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 