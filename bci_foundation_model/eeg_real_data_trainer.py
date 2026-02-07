"""
EEG Next Time Step Prediction Model - Real Neurosity Data
==========================================================
Trains a neural network to predict the next time step's EEG signal
using actual Neurosity Crown 3 EEG recordings.

Dataset: ~11 million samples, 8 channels, 256 Hz
Channels: CP3, C3, F5, PO3, PO4, F6, C4, CP4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# EEG channel names matching Neurosity Crown 3
CHANNELS = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
DATA_PATH = '/Users/jeremynixon/Dropbox/python_new/engineering_consciousness/bci_foundation_model/combined_dataset.csv'


def load_eeg_data(path, n_samples=None):
    """
    Load real EEG data from the Neurosity dataset.

    Parameters:
    - path: Path to combined_dataset.csv
    - n_samples: Number of samples to load (None for all)
    """
    print(f"Loading EEG data from {path}...")

    if n_samples:
        df = pd.read_csv(path, nrows=n_samples)
    else:
        df = pd.read_csv(path)

    print(f"Loaded {len(df):,} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"Sessions: {df['session_id'].nunique()}")

    # Extract EEG channels only
    eeg_data = df[CHANNELS].values.astype(np.float32)
    session_ids = df['session_id'].values

    # Print statistics
    print(f"\nData statistics:")
    for i, ch in enumerate(CHANNELS):
        print(f"  {ch}: mean={eeg_data[:, i].mean():.2f}, std={eeg_data[:, i].std():.2f}")

    return eeg_data, session_ids


def create_sequences(data, session_ids, seq_length=32):
    """
    Create input-output sequences for next-step prediction.
    Only creates sequences within the same session (no cross-session sequences).
    """
    print(f"\nCreating sequences with length {seq_length}...")
    X, y = [], []

    unique_sessions = np.unique(session_ids)
    print(f"Processing {len(unique_sessions)} sessions...")

    for session in unique_sessions:
        session_mask = session_ids == session
        session_data = data[session_mask]

        # Create sequences within this session
        for i in range(len(session_data) - seq_length - 1):
            X.append(session_data[i:i + seq_length])
            y.append(session_data[i + seq_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Created {len(X):,} sequences")
    return X, y


class ConvTransformerModel(nn.Module):
    """
    Convolutional Transformer model for EEG next-step prediction.
    """
    def __init__(self, n_channels=8, seq_length=32, d_model=32, nhead=4, num_layers=1):
        super(ConvTransformerModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=d_model, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear((seq_length // 2) * d_model, n_channels)

    def forward(self, x):
        # x shape: (batch, seq_length, n_channels)
        x = x.permute(0, 2, 1)  # -> (batch, n_channels, seq_length)

        # Convolutions
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Transformer
        x = x.permute(0, 2, 1)  # -> (batch, seq_length//2, d_model)
        x = self.transformer(x)

        # Output
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


def train_model(model, train_loader, valid_loader, device, num_epochs=20, lr=0.001):
    """
    Train the model and track metrics for visualization.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    history = defaultdict(list)

    print("\n" + "="*60)
    print("Beginning Training on Real EEG Data")
    print("="*60)

    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        valid_loss = 0
        valid_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
                valid_batches += 1

                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        avg_valid_loss = valid_loss / valid_batches
        scheduler.step(avg_valid_loss)

        # Compute metrics
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)

        per_channel_mse = np.mean((predictions - targets) ** 2, axis=0)

        # Correlation (handle potential NaN)
        correlations = []
        for i in range(predictions.shape[1]):
            corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        correlation = np.mean(correlations) if correlations else 0

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_valid_loss)
        history['correlation'].append(correlation)
        history['per_channel_mse'].append(per_channel_mse)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), 'best_eeg_model.pth')

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Valid: {avg_valid_loss:.4f} | "
              f"Corr: {correlation:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return history, predictions, targets


def visualize_training(history, predictions, targets, channels, output_prefix='real_data'):
    """
    Create comprehensive visualizations of the training process.
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. Loss curves
    ax1 = fig.add_subplot(2, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['valid_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss (Real EEG Data)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. Correlation over training
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs, history['correlation'], 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_title('Prediction-Target Correlation Over Training', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 3. Per-channel MSE (final epoch)
    ax3 = fig.add_subplot(2, 2, 3)
    final_mse = history['per_channel_mse'][-1]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(channels)))
    bars = ax3.bar(channels, final_mse, color=colors)
    ax3.set_xlabel('EEG Channel', fontsize=12)
    ax3.set_ylabel('MSE', fontsize=12)
    ax3.set_title('Final Per-Channel MSE', fontsize=14)
    ax3.tick_params(axis='x', rotation=45)

    # 4. Predicted vs Actual for sample channel
    ax4 = fig.add_subplot(2, 2, 4)
    sample_idx = 0  # CP3 channel
    n_points = min(300, len(predictions))
    ax4.plot(range(n_points), targets[:n_points, sample_idx], 'b-',
             linewidth=1, label='Actual', alpha=0.7)
    ax4.plot(range(n_points), predictions[:n_points, sample_idx], 'r--',
             linewidth=1, label='Predicted', alpha=0.7)
    ax4.set_xlabel('Time Step', fontsize=12)
    ax4.set_ylabel('Amplitude (μV)', fontsize=12)
    ax4.set_title(f'Predictions vs Actual ({channels[sample_idx]} Channel)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_training_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to '{output_prefix}_training_visualization.png'")

    # Channel comparison plot
    fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (ax, ch) in enumerate(zip(axes, channels)):
        n_points = min(150, len(predictions))
        ax.plot(range(n_points), targets[:n_points, i], 'b-', linewidth=1, label='Actual')
        ax.plot(range(n_points), predictions[:n_points, i], 'r--', linewidth=1, label='Predicted')
        ax.set_title(f'{ch}', fontsize=11)
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Predictions vs Actual - All Channels (Real EEG Data)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_channel_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Channel comparison saved to '{output_prefix}_channel_predictions.png'")


def main():
    print("="*60)
    print("EEG Next Time Step Prediction")
    print("Training on Real Neurosity EEG Data")
    print("="*60)

    # Configuration
    SEQ_LENGTH = 32
    BATCH_SIZE = 128
    NUM_EPOCHS = 15
    N_SAMPLES = 500000  # Use subset for faster training (increase for better results)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load real EEG data
    eeg_data, session_ids = load_eeg_data(DATA_PATH, n_samples=N_SAMPLES)

    # Normalize data (z-score normalization per channel)
    mean = eeg_data.mean(axis=0)
    std = eeg_data.std(axis=0)
    eeg_data_normalized = (eeg_data - mean) / std

    print(f"\nNormalization parameters:")
    for i, ch in enumerate(CHANNELS):
        print(f"  {ch}: mean={mean[i]:.2f}, std={std[i]:.2f}")

    # Create sequences (respecting session boundaries)
    X, y = create_sequences(eeg_data_normalized, session_ids, seq_length=SEQ_LENGTH)

    # Shuffle data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Train/validation split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]

    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_valid):,}")

    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    model = ConvTransformerModel(
        n_channels=len(CHANNELS),
        seq_length=SEQ_LENGTH,
        d_model=32,
        nhead=4,
        num_layers=1
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Train model
    history, predictions, targets = train_model(
        model, train_loader, valid_loader, device,
        num_epochs=NUM_EPOCHS, lr=0.001
    )

    # Final metrics
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['valid_loss'][-1]:.6f}")
    print(f"Final Correlation: {history['correlation'][-1]:.4f}")

    # Visualize results
    print("\nGenerating visualizations...")
    visualize_training(history, predictions, targets, CHANNELS, output_prefix='real_data')

    # Save final model
    model_path = 'eeg_real_data_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalization': {'mean': mean, 'std': std},
        'channels': CHANNELS,
        'seq_length': SEQ_LENGTH,
    }, model_path)
    print(f"\nModel saved to '{model_path}'")


if __name__ == "__main__":
    main()
