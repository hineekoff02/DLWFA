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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ltmodelcr import LSTMTransformerModel

# User configuration
CONFIG = {
    "train_data_folder": "/vols/cms/hin21/delight/waveforms/edata_train",  # Path to the folder containing training HDF5 files
    "val_data_folder": "/vols/cms/hin21/delight/waveforms/edata_val",    # Path to the folder containing validation HDF5 files
    "trace_type": "both",          # "traces_ER", "traces_NR" or "both"
    "use_fourier": True,                # Whether to use Fourier transforms
    "num_epochs": 301,                    # Number of epochs to train
    "checkpoint_folder": "/vols/cms/hin21/delight/waveforms/trial97",  # Folder to save checkpoints
    "loss_folder": "/vols/cms/hin21/delight/waveforms/trial97/loss_data",          # Folder to save loss data and plot
    "start_checkpoint": "/vols/cms/hin21/delight/waveforms/trial97/checkpoint_epoch_288.pth",            # Path to a checkpoint to resume from (or None)
    "batch_size": 256,                   # Batch size for training (events per batch)
    "gradient_clip": 1,               # Maximum gradient norm for clipping
    "learning_rate": 3e-4,              # Learning rate
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
def save_positions_to_csv(true_positions, predicted_positions, true_classes, predicted_classes, true_energies, predicted_energies, epoch, output_folder):
    data = {
        "True_X": true_positions[:, 0],
        "True_Y": true_positions[:, 1],
        "True_Z": true_positions[:, 2],
        "Pred_X": predicted_positions[:, 0],
        "Pred_Y": predicted_positions[:, 1],
        "Pred_Z": predicted_positions[:, 2],
        "True_class": true_classes,
        "Pred_class": predicted_classes,
        "True_nrg": true_energies,
        "Pred_nrg": predicted_energies
    }
    df = pd.DataFrame(data)
    csv_filename = os.path.join(output_folder, f"positions_epoch_{epoch}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Positions and classifications saved to {csv_filename}")

class LazyHDF5Dataset(Dataset):
    def __init__(self, folder, trace_type="both", use_fourier=False):
        """
        Lazy-loading dataset for HDF5 waveforms with event classification and energy regression.

        Parameters:
            folder (str): Path to the folder containing HDF5 files.
            trace_type (str): "traces_ER", "traces_NR", or "both" (loads both and assigns labels).
            use_fourier (bool): Whether to apply Fourier transforms to waveforms.
        """
        self.folder = folder
        self.trace_type = trace_type
        self.use_fourier = use_fourier
        self.raw_paths = sorted(glob.glob(os.path.join(folder, "*.h5")))
        self.strides = [0]
        self.total_events = 0
        self.er_count = 0
        self.nr_count = 0
        self.z_shift = (1700.5 + 1953) / 2  # Center Z-coordinates

        # Normalization factors for energy
        self.energy_min = 50
        self.energy_max = 100000

        self.calculate_offsets()

    def calculate_offsets(self):
        """Calculate cumulative event offsets for multi-file indexing."""
        for path in self.raw_paths:
            with h5py.File(path, "r") as f:
                if self.trace_type == "both":
                    num_er = f["traces_ER"].shape[0]
                    num_nr = f["traces_NR"].shape[0]
                    self.er_count += num_er
                    self.nr_count += num_nr
                    num_events = num_er + num_nr
                else:
                    num_events = f[self.trace_type].shape[0]

                self.strides.append(num_events)
                self.total_events += num_events

        self.strides = np.cumsum(self.strides)

        # Print dataset statistics
        if self.trace_type == "both":
            print(f"Total ER events: {self.er_count}")
            print(f"Total NR events: {self.nr_count}")
        print(f"Total number of events loaded from {len(self.raw_paths)} files: {self.total_events}")

    def __len__(self):
        return self.strides[-1]

    def __getitem__(self, idx):
        """Load a single event by its index."""
        # Find the correct file and local index
        file_idx = np.searchsorted(self.strides, idx, side="right") - 1
        if file_idx < 0 or file_idx >= len(self.raw_paths):
            raise IndexError(f"File index {file_idx} out of bounds!")

        idx_in_file = idx - self.strides[file_idx]

        with h5py.File(self.raw_paths[file_idx], "r") as f:
            if self.trace_type == "both":
                num_er = f["traces_ER"].shape[0]
                num_nr = f["traces_NR"].shape[0]
                total_events = num_er + num_nr

                if idx_in_file >= total_events or idx_in_file < 0:
                    raise IndexError(f"Index {idx_in_file} out of range for file {self.raw_paths[file_idx]}!")

                is_er = idx_in_file < num_er  # First `num_er` events are ER
                trace_type = "traces_ER" if is_er else "traces_NR"
                label_class = 0 if is_er else 1  # ER=0, NR=1

                if not is_er:
                    idx_in_file -= num_er  # Adjust NR index
            else:
                trace_type = self.trace_type
                label_class = 0 if trace_type == "traces_ER" else 1

                num_events_in_file = f[trace_type].shape[0]
                if idx_in_file >= num_events_in_file or idx_in_file < 0:
                    raise IndexError(f"Index {idx_in_file} out of range for file {self.raw_paths[file_idx]} in dataset {trace_type}")

            # Load waveform, position, and energy
            trace = f[trace_type][idx_in_file]
            label = f["positions"][idx_in_file]
            energy = f["energies"][idx_in_file]

            # Center the Z-coordinate
            label[2] += self.z_shift

            # Normalize energy to [0, 1]
            energy = np.log10(energy) / np.log10(100000)

        return (
            torch.tensor(trace, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(label_class, dtype=torch.float32),  # ER=0, NR=1
            torch.tensor(energy, dtype=torch.float32),  # Normalized energy
        )

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = LazyHDF5Dataset(train_data_folder, trace_type=trace_type)
    val_dataset = LazyHDF5Dataset(val_data_folder, trace_type=trace_type)
    print_ram_usage("After data loading")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True)

    # Print dataset statistics
    print(f"Training on {len(train_dataset)} events across {len(train_loader)} batches.")
    print(f"Validating on {len(val_dataset)} events across {len(val_loader)} batches.")

    # Initialize model, loss functions, and optimizer
    model = LSTMTransformerModel().to(device).to(memory_format=torch.channels_last)

    # Loss Functions
    criterion_position = nn.MSELoss()  # For (x, y, z) regression
    criterion_classification = nn.BCEWithLogitsLoss()  # For event type classification
    criterion_energy = nn.MSELoss()  # For energy regression

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=4e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-8)

    # Mixed precision training
    scaler = GradScaler()

    # Load checkpoint if provided
    start_epoch = 1
    if start_checkpoint:
        checkpoint = torch.load(start_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from checkpoint at epoch {start_epoch}")

    # Training loop
    loss_file = os.path.join(loss_folder, "loss_data.csv")

    # Load existing loss data if it exists
    if os.path.exists(loss_file):
        df_existing = pd.read_csv(loss_file)

        last_epoch = df_existing["Epoch"].max()

        start_epoch = last_epoch + 1
        print(f"Resuming from epoch {start_epoch}")
        train_loss_data = []
        val_loss_data = []
    else:
        train_loss_data = []
        val_loss_data = []
        start_epoch = 1  # Start from scratch if no file exists

    true_positions, predicted_positions = [], []
    true_classes, predicted_classes = [], []
    true_energies, predicted_energies = [], []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training", dynamic_ncols=True) as t:
            for batch_idx, (waveforms, positions, labels, energies) in enumerate(t):
                batch_size = positions.size(0)
                tqdm.write(f"Processing batch {batch_idx + 1}/{len(train_loader)} with {batch_size} events.")

                # Move data to GPU
                waveforms, positions, labels, energies = (
                    waveforms.to(device), 
                    positions.to(device), 
                    labels.to(device), 
                    energies.to(device)
                )

                optimizer.zero_grad()

                # Forward pass
                with autocast():
                    pred_positions, pred_classes, pred_energies = model(waveforms)
                    loss_position = criterion_position(pred_positions, positions)
                    loss_classification = criterion_classification(pred_classes.squeeze(), labels)
                    loss_energy = criterion_energy(pred_energies.squeeze(), energies)

                    # Total loss (adjust weights if needed)
                    loss = loss_position + loss_classification + loss_energy  

                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                t.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_data.append(avg_train_loss)
        print_ram_usage(f"After epoch {epoch}")

        # Validation loop
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Validation", dynamic_ncols=True) as t:
                for batch_idx, (waveforms, positions, labels, energies) in enumerate(t):
                    batch_size = positions.size(0)
                    tqdm.write(f"Processing batch {batch_idx + 1}/{len(val_loader)} with {batch_size} events.")

                    waveforms, positions, labels, energies = (
                        waveforms.to(device), 
                        positions.to(device), 
                        labels.to(device), 
                        energies.to(device)
                    )

                    # Forward pass
                    with autocast():
                        pred_positions, pred_classes, pred_energies = model(waveforms)
                        loss_position = criterion_position(pred_positions, positions)
                        loss_classification = criterion_classification(pred_classes.squeeze(), labels)
                        loss_energy = criterion_energy(pred_energies.squeeze(), energies)
                        loss = loss_position + loss_classification + loss_energy  

                    total_val_loss += loss.item()
                    t.set_postfix(loss=loss.item())

                    # Append results
                    true_positions.append(positions.cpu().numpy())
                    predicted_positions.append(pred_positions.cpu().numpy())
                    true_classes.append(labels.cpu().numpy())
                    predicted_classes.append(pred_classes.cpu().numpy())
                    true_energies.append(energies.cpu().numpy())
                    predicted_energies.append(pred_energies.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_data.append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: Learning Rate = {current_lr:.6f}")

        # Scheduler step
        scheduler.step()

        # Flatten lists into arrays for saving
        true_positions = np.concatenate(true_positions, axis=0)
        predicted_positions = np.concatenate(predicted_positions, axis=0)
        true_classes = np.concatenate(true_classes, axis=0).flatten()
        predicted_classes = np.concatenate(predicted_classes, axis=0).flatten()
        true_energies = np.concatenate(true_energies, axis=0).flatten()
        predicted_energies = np.concatenate(predicted_energies, axis=0).flatten()

        # Save data to CSV
        save_positions_to_csv(true_positions, predicted_positions, true_classes, predicted_classes, 
                              true_energies, predicted_energies, epoch + 1, checkpoint_folder)

        # Reset lists for next epoch
        true_positions, predicted_positions = [], []
        true_classes, predicted_classes = [], []
        true_energies, predicted_energies = [], []

        # Print resource usage after validation
        print_ram_usage(f"Epoch {epoch} - Post-Validation RAM Usage")

        print(f"Epoch {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Save loss data
        if os.path.exists(loss_file):
            df_existing = pd.read_csv(loss_file)

            # Get last recorded epoch
            if not df_existing.empty:
                last_epoch = df_existing["Epoch"].max()
            else:
                last_epoch = 0  # No previous epochs found
        else:
            df_existing = None
            last_epoch = 0  # No previous file exists, start fresh

        print(f"Resuming from epoch {last_epoch}")

        # Capture only the current epoch's loss values
        df_new = pd.DataFrame({
            "Epoch": [epoch],  # Save only the current epoch number
            "Train Loss": [avg_train_loss],  # Save only the latest loss
            "Validation Loss": [avg_val_loss]
        })

        # Append only new data instead of saving everything again
        if df_existing is not None:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new  # First time creating file

        df_combined.to_csv(loss_file, index=False)

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

