import subprocess
import json
import pandas as pd
import os
import time
import datetime
import tkinter as tk
from tkinter import ttk
import threading
import random
from typing import List, Dict, Any

class NetworkTrafficSimulator:
    def __init__(self, root, input_directory):
        self.root = root
        self.root.title("ICS Network Traffic Simulator")
        self.root.geometry("1200x800")
        
        self.input_directory = input_directory
        self.csv_files = self.get_csv_files()
        self.current_file_index = 0
        self.current_row = 0
        self.data = None
        self.is_running = False
        self.speed_factor = 1.0
        self.simulation_start_time = None
        self.anomaly_detection_active = False
        
        # Add path to anomaly detector script
        self.anomaly_detector_path = "anomaly_detector.py"
        self.setup_ui()
        self.load_next_file()

    def check_for_anomaly(self, row_data):
        """Check if the current data point is an anomaly using external script"""
        if not self.anomaly_detection_active:
            return False
        
        try:
            # Prepare data as comma-separated string
            data_str = ",".join(str(value) for value in row_data)
            
            # Call anomaly_detector.py as a subprocess
            process = subprocess.Popen(
                ['python', self.anomaly_detector_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send data to the process
            stdout, stderr = process.communicate(input=data_str)
            
            if stderr:
                print(f"Error from anomaly detector: {stderr}")
                return False
            
            # Parse the output
            if "No anomaly detected" in stdout:
                print("Packet OK - No anomaly detected")
                return False
            elif "ANOMALY DETECTED" in stdout:
                print("⚠️ Anomaly detected in packet!")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return False

    def update_display(self, row_data):
        """Update the display with the modified row data"""
        try:
            values = row_data.tolist()
            
            # Determine if this is an anomaly using the external detector
            is_anomaly = False
            if self.anomaly_detection_active:
                is_anomaly = self.check_for_anomaly(values)
            
            # Insert the row with appropriate tag
            item_id = self.tree.insert("", tk.END, values=values)
            
            # Apply tags if anomaly detection is active
            if self.anomaly_detection_active:
                if is_anomaly:
                    self.tree.item(item_id, tags=('anomaly',))
                    self.status_var.set("⚠️ Anomaly detected in packet!")
                else:
                    self.tree.item(item_id, tags=('normal',))
                    self.status_var.set("✅ Packet OK")
            
            # Keep only last 100 rows visible
            if len(self.tree.get_children()) > 100:
                first_item = self.tree.get_children()[0]
                self.tree.delete(first_item)
            
            # Ensure latest entry is visible
            self.tree.yview_moveto(1.0)
            self.root.update_idletasks()
        except Exception as e:
            print(f"Error updating display: {str(e)}")

    def toggle_anomaly_detection(self):
        """Toggle anomaly detection on/off"""
        self.anomaly_detection_active = not self.anomaly_detection_active
        if self.anomaly_detection_active:
            self.anomaly_status_var.set("On")
            self.anomaly_button.config(text="Disable Anomaly Detection")
            self.status_var.set("Anomaly detection enabled")
        else:
            self.anomaly_status_var.set("Off")
            self.anomaly_button.config(text="Enable Anomaly Detection")
            self.status_var.set("Anomaly detection disabled")

    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=10)
        
        # Directory info
        ttk.Label(control_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(control_frame, text=self.input_directory).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Current file info
        ttk.Label(control_frame, text="Current File:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.file_var = tk.StringVar(value="None")
        ttk.Label(control_frame, textvariable=self.file_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Speed control
        ttk.Label(control_frame, text="Speed:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(control_frame, from_=0.1, to=10.0, variable=self.speed_var, orient=tk.HORIZONTAL, length=200)
        speed_scale.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(control_frame, textvariable=self.speed_var).grid(row=2, column=2, padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Anomaly Detection button
        self.anomaly_button = ttk.Button(button_frame, text="Enable Anomaly Detection", command=self.toggle_anomaly_detection)
        self.anomaly_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(status_frame, text="Current Time:").pack(side=tk.LEFT, padx=(20, 5))
        self.time_var = tk.StringVar(value="--:--:--")
        self.date_var = tk.StringVar(value="----/--/--")
        ttk.Label(status_frame, textvariable=self.time_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=self.date_var).pack(side=tk.LEFT, padx=5)
        
        # Anomaly status
        ttk.Label(status_frame, text="Anomaly Detection:").pack(side=tk.LEFT, padx=(20, 5))
        self.anomaly_status_var = tk.StringVar(value="Off")
        ttk.Label(status_frame, textvariable=self.anomaly_status_var).pack(side=tk.LEFT, padx=5)
        
        # Traffic display
        traffic_frame = ttk.LabelFrame(main_frame, text="Network Traffic", padding="10")
        traffic_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create treeview for traffic display with scrollbars
        self.create_treeview(traffic_frame)
        
        # Configure tag styles for anomaly detection
        self.tree.tag_configure('normal', background='light green')
        self.tree.tag_configure('anomaly', background='red')

    def create_treeview(self, parent_frame):
        """Create the treeview with scrollbars"""
        # Create a frame for the treeview and scrollbars
        tree_frame = ttk.Frame(parent_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Create the treeview
        columns = ["num", "date", "time", "orig", "type", "i/f_name", "i/f_dir", "src", "dst", 
                  "proto", "appi_name", "proxy_src_ip", "Modbus_Function_Code", 
                  "Modbus_Function_Description", "Modbus_Transaction_ID", "SCADA_Tag", 
                  "Modbus_Value", "service", "s_port", "Tag"]
        
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        # Set column headings and widths
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        # Create scrollbars
        y_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        
        # Configure treeview scrolling
        self.tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)

        # Grid layout for treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")

        # Configure grid weights
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

    def get_csv_files(self):
        """Get list of CSV files from the input directory"""
        if not os.path.exists(self.input_directory):
            raise FileNotFoundError(f"Input directory not found: {self.input_directory}")
        
        csv_files = [f for f in os.listdir(self.input_directory) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_directory}")
        
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
        return sorted(csv_files)

    def load_next_file(self):
        """Load the next CSV file from the input directory"""
        if self.current_file_index >= len(self.csv_files):
            self.status_var.set("All files processed")
            return False
        
        file_name = self.csv_files[self.current_file_index]
        file_path = os.path.join(self.input_directory, file_name)
        self.file_var.set(file_name)
        
        try:
            self.data = pd.read_csv(file_path)
            print(f"Successfully loaded {file_name} with {len(self.data)} records")
            self.status_var.set(f"Loaded {len(self.data)} records from {file_name}")
            self.current_row = 0
            self.current_file_index += 1
            return True
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
            self.status_var.set(f"Error loading file {file_name}: {str(e)}")
            self.current_file_index += 1
            return self.load_next_file()

    def start_simulation(self):
        """Start the simulation"""
        if self.data is None or self.data.empty:
            self.status_var.set("No data loaded.")
            return
        
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start simulation in a separate thread
            threading.Thread(target=self.run_simulation, daemon=True).start()

    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Simulation paused")

    def reset_simulation(self):
        """Reset the simulation"""
        self.stop_simulation()
        self.current_file_index = 0
        self.current_row = 0
        self.simulation_start_time = None
        
        # Clear the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.load_next_file()
        self.status_var.set("Simulation reset")
        self.time_var.set("--:--:--")
        self.date_var.set("----/--/--")

    def run_simulation(self):
        """Run the simulation"""
        if not self.simulation_start_time:
            self.simulation_start_time = datetime.datetime.now()
            print(f"Starting simulation at: {self.simulation_start_time}")
        
        while self.is_running:
            if self.current_row >= len(self.data):
                if not self.load_next_file():
                    print("Finished processing all files")
                    self.status_var.set("Simulation completed - all files processed")
                    self.stop_simulation()
                    break
            
            try:
                # Get current row data
                row_data = self.data.iloc[self.current_row]
                
                # Update timestamp
                current_time = datetime.datetime.now()
                self.time_var.set(current_time.strftime("%H:%M:%S"))
                self.date_var.set(current_time.strftime("%Y-%m-%d"))
                
                # Update display
                self.update_display(row_data)
                
                self.current_row += 1
                time.sleep(0.1 / self.speed_factor)
                
            except Exception as e:
                print(f"Error in simulation: {e}")
                continue

if __name__ == "__main__":
    # Update the input directory path
    INPUT_DIRECTORY = "Dataset"  # This will look for Dataset/preprocessed_batch_1.csv
    
    root = tk.Tk()
    app = NetworkTrafficSimulator(root, INPUT_DIRECTORY)
    root.mainloop() 