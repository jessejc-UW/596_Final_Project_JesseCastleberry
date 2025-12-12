import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend
import torch
from torch.utils.data import DataLoader, TensorDataset
from .model import *
import copy
import matplotlib.pyplot as plt

def load_data(file_path):
    # Load the dataset from a .npz file
    data = np.load(file_path, allow_pickle=True)
    X = data['radar']
    y = data['hr_bpm']

    # Construct pairs of [input, target]
    return [[X[i], y[i]] for i in range(len(y))]

def scale_data(data, scaler=None):
    # Ensure we are working with a simple list of (input, target) pairs
    samples = list(data)
    X = np.stack([np.array(pair[0]) for pair in samples])
    y = np.array([pair[1] for pair in samples])

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return [[X_scaled[i], y[i]] for i in range(len(y))], scaler

    X_scaled = scaler.transform(X)
    return [[X_scaled[i], y[i]] for i in range(len(y))]
    
def detrend_input(data):
    # Detrend the input signal
    for i, pair in enumerate(data):
        data[i][0] = detrend(pair[0])
    return data

def split_data(data, train_ratio=0.8, shuffle=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    # Shuffle data if required
    if shuffle:
        np.random.shuffle(data)
    
    # Split data into training and validation sets
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data

def create_dataloader(samples, batch_size=32, shuffle=True):
    X = np.array([sample[0] for sample in samples])
    y = np.array([sample[1] for sample in samples])
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def prepare_data(data_path, test_path, seed=14):
    # Load and preprocess training data
    train_data = load_data(data_path)
    train_data = detrend_input(train_data)
    train_data, val_data = split_data(train_data, seed=seed, shuffle=True)
    train_data, scaler = scale_data(train_data)
    val_data = scale_data(val_data, scaler=scaler)

    # Load and preprocess test data
    test_data = load_data(test_path)
    test_data = detrend_input(test_data)
    test_data = scale_data(test_data, scaler=scaler)

    train_loader = create_dataloader(train_data, batch_size=32, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=32, shuffle=False)
    test_loader = create_dataloader(test_data, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, scaler

def init_model():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    print(f'Using {device} for training')

    model = HeartRateEstimator().to(device)
    print(model)
    return model, device

def train_model(model, train_loader, val_loader, epochs=500, lr=5e-5, batch_size=32, save_model=False, device='cpu', plot_loss=True, verbose=True, checkpoint_dir='../checkpoints/', results_dir='../results/'):
    lr = lr
    batch_size=batch_size
    
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

    train_loss_list = []
    val_loss_list = []

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1
    patience = 100
    epochs_no_improve = 0
    min_delta = 0.0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=int(patience*0.75)
    )

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1)  # Add channel dimension

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_list.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(1)  # Add channel dimension

                outputs = model(inputs).squeeze()
                loss = loss_fn(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_list.append(epoch_val_loss)

        if verbose: print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step(epoch_val_loss)

        # Early stopping / best model tracking
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose: print(f"Early stopping triggered at epoch {epoch+1}. "
                    f"Best epoch was {best_epoch+1} with val loss {best_val_loss:.4f}.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if verbose: print(f"Loaded best model from epoch {best_epoch+1} "
            f"(val loss = {best_val_loss:.4f})")
        
    metrics = {'train_loss': train_loss_list,
               'val_loss': val_loss_list,
               }
        
    if save_model:
        model.save_model(save_dir=checkpoint_dir)
        # save training metrics
        os.makedirs(results_dir, exist_ok=True)
        metrics_path = os.path.join(results_dir, 'training_metrics.npz')
        np.savez(metrics_path, train_loss=train_loss_list, val_loss=val_loss_list)

    if plot_loss:
        # Plot the losses
        figure = plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
        plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        # save figure
        if save_model:
            fig_path = os.path.join(results_dir, 'training_loss_curve.png')
            figure.savefig(fig_path)
            plt.show(block=False)

    return model, metrics

def evaluate_model(model, test_loader, train_loader, device='cpu', results_dir=None, verbose=False):

    # Plot the predicted vs actual heart rates for test set
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device).unsqueeze(1)  # Add channel dimension
            outputs = model(inputs).squeeze().cpu().numpy()
            test_preds.extend(outputs)
            test_targets.extend(targets.numpy())

    # Add estimations from training data as well for comparison
    train_preds = []
    train_targets = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device).unsqueeze(1)  # Add channel dimension
            outputs = model(inputs).squeeze().cpu().numpy()
            train_preds.extend(outputs)
            train_targets.extend(targets.numpy())
    figure = plt.figure(figsize=(8, 8))
    plt.scatter(train_targets, train_preds, alpha=0.3, label='Train Data', color='orange')
    plt.scatter(test_targets, test_preds, alpha=0.7, label='Test Data', color='blue')
    plt.plot([40, 180], [40, 180], 'r--')
    plt.xlim(40, 180)
    plt.ylim(40, 180)
    plt.xlabel('Actual Heart Rate (BPM)')
    plt.ylabel('Estimated Heart Rate (BPM)')
    plt.title('Estimated vs Actual Heart Rates')
    plt.legend()

    # Calculate regression metrics for test set
    ss_res = np.sum((np.array(test_targets) - np.array(test_preds)) ** 2)
    ss_tot = np.sum((np.array(test_targets) - np.mean(test_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    if verbose: print(f"Test R^2: {r2:.4f}")
    # Calculate MAE for test set
    mae = np.mean(np.abs(np.array(test_targets) - np.array(test_preds)))
    if verbose: print(f"Test MAE: {mae:.4f} BPM")
    # Calculate regression metrics for training set
    ss_res_train = np.sum((np.array(train_targets) - np.array(train_preds)) ** 2)
    ss_tot_train = np.sum((np.array(train_targets) - np.mean(train_targets)) ** 2)
    r2_train = 1 - (ss_res_train / ss_tot_train)
    if verbose: print(f"Train R^2: {r2_train:.4f}")
    # Calculate MAE for training set
    mae_train = np.mean(np.abs(np.array(train_targets) - np.array(train_preds)))
    if verbose: print(f"Train MAE: {mae_train:.4f} BPM")

    # add MAE and regression info to existing plot
    # set location to bottom right
    plt.figtext(0.6, 0.2, f'Test R²: {r2:.4f}\nTest MAE: {mae:.2f} BPM\n\nTrain R²: {r2_train:.4f}\nTrain MAE: {mae_train:.2f} BPM',
                bbox=dict(facecolor='white', alpha=0.5))

    # save figure if directory given
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        fig_path = os.path.join(results_dir, 'performance.png')
        figure.savefig(fig_path)
    plt.show()