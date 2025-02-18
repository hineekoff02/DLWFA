import pandas as pd
import psutil
import h5py
import numpy as np
import glob
import os
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from ltmodel2 import LSTMTransformerModel

# User configuration
CONFIG = {
    "train_data_folder": "/vols/cms/hin21/delight/waveforms/data_train",  # Path to the folder containing training HDF5 files
    "val_data_folder": "/vols/cms/hin21/delight/waveforms/data_val",    # Path to the folder containing validation HDF5 files
    "trace_type": "traces_ER",          # "traces_ER" or "traces_NR"
    "use_fourier": True,                # Whether to use Fourier transforms
    "num_epochs": 300,                    # Number of epochs to train
    "checkpoint_folder": "/vols/cms/hin21/delight/waveforms/trial39",  # Folder to save checkpoints
    "loss_folder": "/vols/cms/hin21/delight/waveforms/trial39/loss_data",          # Folder to save loss data and plot
    "start_checkpoint": None,            # Path to a checkpoint to resume from (or None)
    "batch_size": 128,                   # Batch size for training (events per batch)
    "gradient_clip": 1.0,               # Maximum gradient norm for clipping
    "learning_rate": 2.5e-4,              # Learning rate
    "scheduler_step_size": 25,  # Decay learning rate every X epochs
    "scheduler_gamma": 0.5,    # Decay factor for scheduler
}

#device = torch.cuda.current_device()  # Gets the active GPU
#torch.cuda.set_per_process_memory_fraction(0.7, device=0)  # Change "0" to another GPU if needed

#torch.backends.cudnn.allow_tf32 = False  
#torch.backends.cuda.matmul.allow_tf32 = False

# Ram usage function
def print_ram_usage(tag=""):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{tag} RAM used: {mem_info.rss / 1e9:.2f} GB")  # RSS: Resident Set Size

# Function to apply FFT and normalize
def apply_fft(trace):
    fft_result = np.fft.fft(trace, axis=-1)
    fft_magnitude = np.abs(fft_result)  # Use magnitude
    fft_magnitude = (fft_magnitude - np.mean(fft_magnitude)) / (np.std(fft_magnitude) + 1e-6)
    return fft_magnitude

# Function to save positions to CSV
def save_positions_to_csv(true_positions, predicted_positions, epoch, energy, output_folder):
    data = {
        "True_X": true_positions[:, 0],
        "True_Y": true_positions[:, 1],
        "True_Z": true_positions[:, 2],
        "Pred_X": predicted_positions[:, 0],
        "Pred_Y": predicted_positions[:, 1],
        "Pred_Z": predicted_positions[:, 2],
        "Energy": energy
    }
    df = pd.DataFrame(data)
    csv_filename = os.path.join(output_folder, f"positions_epoch_{epoch}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Positions saved to {csv_filename}")

class LazyHDF5Dataset(Dataset):
    def __init__(self, folder, trace_type='traces_ER', use_fourier=False):
        """
        Initialize the dataset to load events lazily from multiple HDF5 files.

        Parameters:
            folder (str): Path to the folder containing HDF5 files.
            trace_type (str): Type of traces to load (e.g., 'traces_ER').
            use_fourier (bool): Whether to apply Fourier transforms to waveforms.
        """
        self.folder = folder
        self.trace_type = trace_type
        self.use_fourier = use_fourier
        self.raw_paths = sorted(glob.glob(os.path.join(folder, '*.h5')))
        self.strides = [0]
        self.total_events = 0
        self.calculate_offsets()

    def calculate_offsets(self):
        """Calculate cumulative offsets for all files."""
        for path in self.raw_paths:
            with h5py.File(path, 'r') as f:
                num_events = f[self.trace_type].shape[0]
                self.strides.append(num_events)
                self.total_events += num_events
        self.strides = np.cumsum(self.strides)
        print(f"Total number of events loaded from {len(self.raw_paths)} files: {self.total_events}")

    def __len__(self):
        return self.strides[-1]

    def __getitem__(self, idx):
        """Load a single event by its index."""
        file_idx = np.searchsorted(self.strides, idx, side='right') - 1
        idx_in_file = idx - self.strides[file_idx]

        with h5py.File(self.raw_paths[file_idx], 'r') as f:
            trace = f[self.trace_type][idx_in_file]
            label = f['positions'][idx_in_file]

        return torch.tensor(trace, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Training Script
if __name__ == "__main__":
    # Load configuration
    train_data_folder = CONFIG["train_data_folder"]
    val_data_folder = CONFIG["val_data_folder"]
    trace_type = CONFIG["trace_type"]
    use_fourier = CONFIG["use_fourier"]
    num_epochs = CONFIG["num_epochs"]
    checkpoint_folder = CONFIG["checkpoint_folder"]
    loss_folder = CONFIG["loss_folder"]
    start_checkpoint = CONFIG["start_checkpoint"]
    batch_size = CONFIG["batch_size"]
    gradient_clip = CONFIG["gradient_clip"]
    lr = CONFIG["learning_rate"]
    scheduler_step_size = CONFIG["scheduler_step_size"]
    scheduler_gamma = CONFIG["scheduler_gamma"]

    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(loss_folder, exist_ok=True)

    # Check for CUDA device
   # device = torch.device("cpu")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = LazyHDF5Dataset(train_data_folder, trace_type=trace_type)
    val_dataset = LazyHDF5Dataset(val_data_folder, trace_type=trace_type)
    print_ram_usage("After data loading")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers = 6, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 6, pin_memory = True)

    # Print the number of events loaded
    print(f"Training on {len(train_dataset)} events across {len(train_loader)} batches.")
    print(f"Validating on {len(val_dataset)} events across {len(val_loader)} batches.")

    # Initialize model, loss, and optimizer
    model = LSTMTransformerModel().to(device)
    #print(f"Allocated GPU memory after model init: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    #print(f"Reserved GPU memory after model init: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Mixed precision training
    scaler = GradScaler() #init_scale = 512

    # Load checkpoint if provided
    start_epoch = 0
    if start_checkpoint:
        checkpoint = torch.load(start_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from checkpoint at epoch {start_epoch}")

    # Training loop
    train_loss_data = []
    val_loss_data = []

    true_positions = []
    predicted_positions = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0
        #torch.cuda.empty_cache()
        #torch.cuda.reset_peak_memory_stats()  # Free memory before training starts
        #torch.cuda.ipc_collect()  # Collects unused memory
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training") as t:
            for batch_idx, (waveforms, positions) in enumerate(t):
                batch_size = positions.size(0)  # Get the number of events in the current batch
                print(f"Processing batch {batch_idx + 1}/{len(train_loader)} with {batch_size} events.")
                waveforms, positions = waveforms.to(device), positions.to(device)

                optimizer.zero_grad()

                with autocast(): #dtype=torch.float16):
                    outputs = model(waveforms)
                    loss = criterion(outputs, positions)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = gradient_clip) # Gradient clipping
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                t.set_postfix(loss=loss.item())
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_data.append(avg_train_loss)
     #   print(f"Epoch {epoch + 1} - Post-Training GPU Memory Usage:")
      #  print(f"Allocated GPU memory after epoch {epoch + 1}: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
       # print(f"Reserved GPU memory after epoch {epoch + 1}: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        print_ram_usage(f"After epoch {epoch + 1}")

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation") as t:
                for batch_idx, (waveforms, positions) in enumerate(t):
                    batch_size = positions.size(0)  # Get the number of events in the current batch
                    print(f"Processing batch {batch_idx + 1}/{len(val_loader)} with {batch_size} events.")
                    waveforms, positions = waveforms.to(device), positions.to(device)

                    with autocast(): #dtype=torch.float16):
                        outputs = model(waveforms)
                        loss = criterion(outputs, positions)
                    total_val_loss += loss.item()
                    t.set_postfix(loss=loss.item())
                    # Ensure true_positions is a list before appending
                    if not isinstance(true_positions, list):
                        true_positions = list(true_positions)
                    if not isinstance(predicted_positions, list):
                        predicted_positions = list(predicted_positions)

                    # Append true and predicted positions
                    true_positions.append(positions.cpu().numpy())
                    predicted_positions.append(outputs.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_data.append(avg_val_loss)

        # Scheduler step
        scheduler.step()

        # Flatten the lists into arrays for CSV saving
        true_positions = np.concatenate(true_positions, axis=0)
        predicted_positions = np.concatenate(predicted_positions, axis=0)

        # Save the true and predicted positions to CSV
        save_positions_to_csv(true_positions, predicted_positions, epoch + 1, 10000, checkpoint_folder)

        # Reinitialize true_positions and predicted_positions as lists for the next epoch
        true_positions = []
        predicted_positions = []

        # Print resource usage after validation phase
        #print(f"Epoch {epoch + 1} - Post-Validation GPU Memory Usage:")
        #print(f"  Allocated GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        #print(f"  Reserved GPU memory: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        print_ram_usage(f"Epoch {epoch + 1} - Post-Validation RAM Usage")

        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        # Convert lists to DataFrame
        df = pd.DataFrame({"Epoch": list(range(1, len(train_loss_data) + 1)), 
                       "Train Loss": train_loss_data, 
                       "Validation Loss": val_loss_data})

        # Save to CSV
        df.to_csv(os.path.join(loss_folder, "loss_data.csv"), index=False)
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

