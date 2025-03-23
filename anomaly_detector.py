import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configuration
model_path = "/home/smartdragon/Videos/SWAT/SWaT.A1 & A2_Dec 2015/Data/Results/autoencoder_progressive.h5"
scaler_path = "/home/smartdragon/Videos/SWAT/SWaT.A1 & A2_Dec 2015/Data/Results/scaler.pkl"
ANOMALY_THRESHOLD = 0.01  # Adjust this based on your needs

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_model_and_scaler(model_path, scaler_path, input_dim, device="cpu"):
    """Loads the autoencoder model and scaler."""
    try:
        model = Autoencoder(input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        scaler = pd.read_pickle(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def detect_anomaly(model, scaler, input_data, device="cpu"):
    """Detects anomalies in the input data."""
    try:
        model.eval()
        with torch.no_grad():
            # Scale the input data
            if isinstance(input_data, list):
                input_data = np.array(input_data).reshape(1, -1)
            scaled_data = scaler.transform(input_data)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(scaled_data).to(device)
            
            # Get reconstruction
            reconstructed = model(input_tensor)
            
            # Calculate reconstruction error
            error = torch.mean((input_tensor - reconstructed) ** 2).item()
            
            # Determine if it's an anomaly
            is_anomaly = error > ANOMALY_THRESHOLD
            
            return is_anomaly, error
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return False, 0.0

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 121  # Replace with the actual input dimension

    # Load the model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path, input_dim, device)
    
    if model is None or scaler is None:
        print("Error: Failed to load model or scaler")
        return

    try:
        # Get input from stdin
        input_str = input().strip()
        input_list = [float(x.strip()) for x in input_str.split(",")]
        
        if len(input_list) != input_dim:
            print(f"Error: Input must have {input_dim} values.")
            return
        
        # Detect anomaly
        is_anomaly, reconstruction_error = detect_anomaly(model, scaler, input_list, device)

        # Print result in a format that can be parsed by the main program
        if is_anomaly:
            print("ANOMALY DETECTED")
        else:
            print("No anomaly detected")
        print(f"Reconstruction Error: {reconstruction_error:.6f}")
        print(f"Threshold: {ANOMALY_THRESHOLD:.6f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()