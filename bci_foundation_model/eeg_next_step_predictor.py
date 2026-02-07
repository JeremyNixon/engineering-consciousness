"""
EEG Next Time Step Prediction Model
====================================
Trains a neural network to predict the next time step's EEG signal
using a ConvTransformer architecture.

Since the Dropbox link requires authentication, we generate synthetic
EEG-like data matching the Neurosity Crown 3 specifications:
- 8 channels: CP3, C3, F5, PO3, PO4, F6, C4, CP4
- 256 Hz sampling rate
- Realistic EEG frequency components (alpha, beta, theta, delta bands)
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
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# EEG channel names matching Neurosity Crown 3
CHANNELS = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
SAMPLING_RATE = 256  # Hz


def generate_synthetic_eeg(n_samples=50000, n_channels=8, fs=256):
    """
    Generate synthetic EEG-like signals with realistic frequency components.

    EEG signals contain:
    - Delta (0.5-4 Hz): Deep sleep
    - Theta (4-8 Hz): Drowsiness, light sleep
    - Alpha (8-13 Hz): Relaxed, eyes closed
    - Beta (13-30 Hz): Active thinking, focus
    - Gamma (30-100 Hz): Higher cognitive functions
    """
    print("Generating synthetic EEG data...")
    t = np.arange(n_samples) / fs
    data = np.zeros((n_samples, n_channels))

    for ch in range(n_channels):
        # Base signal with different frequency bands
        # Delta band (0.5-4 Hz)
        delta = 30 * np.sin(2 * np.pi * (1.5 + 0.3 * ch) * t + np.random.random() * 2 * np.pi)

        # Theta band (4-8 Hz)
        theta = 20 * np.sin(2 * np.pi * (6 + 0.2 * ch) * t + np.random.random() * 2 * np.pi)

        # Alpha band (8-13 Hz) - dominant in relaxed state
        alpha = 25 * np.sin(2 * np.pi * (10 + 0.5 * ch) * t + np.random.random() * 2 * np.pi)

        # Beta band (13-30 Hz)
        beta = 10 * np.sin(2 * np.pi * (20 + ch) * t + np.random.random() * 2 * np.pi)

        # Gamma band (30-100 Hz) - smaller amplitude
        gamma = 5 * np.sin(2 * np.pi * (40 + 2 * ch) * t + np.random.random() * 2 * np.pi)

        # Add pink noise (1/f noise characteristic of EEG)
        white_noise = np.random.randn(n_samples)
        # Simple pink noise approximation
        pink_noise = np.cumsum(white_noise) / np.sqrt(n_samples) * 15

        # Combine all components
        data[:, ch] = delta + theta + alpha + beta + gamma + pink_noise

        # Add some channel-specific variation
        data[:, ch] *= (0.8 + 0.4 * np.random.random())

    print(f"Generated {n_samples} samples across {n_channels} channels")
    return data


def create_sequences(data, seq_length=32):
    """
    Create input-output sequences for next-step prediction.

    Input: sequence of seq_length time steps (shape: seq_length x n_channels)
    Output: the next time step (shape: n_channels)
    """
    print(f"Creating sequences with length {seq_length}...")
    X, y = [], []

    for i in range(len(data) - seq_length - 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Created {len(X)} sequences")
    return X, y


class ConvTransformerModel(nn.Module):
    """
    Convolutional Transformer model for EEG next-step prediction.

    Architecture:
    1. 1D Convolutions to extract local temporal features
    2. Transformer encoder for global temporal dependencies
    3. Fully connected layer for final prediction
    """
    def __init__(self, n_channels=8, seq_length=32, d_model=32, nhead=4, num_layers=1):
        super(ConvTransformerModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=d_model, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

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
        # After pooling: seq_length // 2 = 16 time steps, d_model features each
        self.fc = nn.Linear((seq_length // 2) * d_model, n_channels)

    def forward(self, x):
        # x shape: (batch, seq_length, n_channels)
        x = x.permute(0, 2, 1)  # -> (batch, n_channels, seq_length)

        # Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # -> (batch, d_model, seq_length//2)

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
    criterion = nn.MSELoss()

    history = defaultdict(list)

    print("\n" + "="*60)
    print("Beginning Training")
    print("="*60)

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

        # Compute per-channel metrics
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)

        per_channel_mse = np.mean((predictions - targets) ** 2, axis=0)
        correlation = np.mean([np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
                              for i in range(predictions.shape[1])])

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_valid_loss)
        history['correlation'].append(correlation)
        history['per_channel_mse'].append(per_channel_mse)

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Valid Loss: {avg_valid_loss:.6f} | "
              f"Correlation: {correlation:.4f}")

    return history, predictions, targets


def visualize_training(history, predictions, targets, channels):
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
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Correlation over training
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs, history['correlation'], 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_title('Prediction-Target Correlation Over Training', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

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
    n_points = min(200, len(predictions))
    ax4.plot(range(n_points), targets[:n_points, sample_idx], 'b-',
             linewidth=1.5, label='Actual', alpha=0.7)
    ax4.plot(range(n_points), predictions[:n_points, sample_idx], 'r--',
             linewidth=1.5, label='Predicted', alpha=0.7)
    ax4.set_xlabel('Time Step', fontsize=12)
    ax4.set_ylabel('Amplitude', fontsize=12)
    ax4.set_title(f'Predictions vs Actual ({channels[sample_idx]} Channel)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nVisualization saved to 'training_visualization.png'")

    # Additional detailed channel comparison plot
    fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (ax, ch) in enumerate(zip(axes, channels)):
        n_points = min(100, len(predictions))
        ax.plot(range(n_points), targets[:n_points, i], 'b-', linewidth=1, label='Actual')
        ax.plot(range(n_points), predictions[:n_points, i], 'r--', linewidth=1, label='Predicted')
        ax.set_title(f'{ch}', fontsize=11)
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Predictions vs Actual - All Channels', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('channel_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Channel comparison saved to 'channel_predictions.png'")


def main():
    print("="*60)
    print("EEG Next Time Step Prediction")
    print("ConvTransformer Architecture")
    print("="*60)

    # Configuration
    SEQ_LENGTH = 32
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    N_SAMPLES = 50000  # Synthetic data size

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Generate synthetic EEG data
    eeg_data = generate_synthetic_eeg(n_samples=N_SAMPLES, n_channels=len(CHANNELS))

    # Normalize data (z-score normalization per channel)
    mean = eeg_data.mean(axis=0)
    std = eeg_data.std(axis=0)
    eeg_data = (eeg_data - mean) / std

    # Create sequences
    X, y = create_sequences(eeg_data, seq_length=SEQ_LENGTH)

    # Train/validation split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")

    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    visualize_training(history, predictions, targets, CHANNELS)

    # Save model
    model_path = 'eeg_predictor_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to '{model_path}'")


if __name__ == "__main__":
    main()
