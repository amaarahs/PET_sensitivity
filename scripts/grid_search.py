import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import matplotlib.pyplot as plt
#import numpy as np
import os
import sys
import pandas as pd
from pathlib import Path
from sirf.STIR import set_verbosity
from itertools import product

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
source_path = os.path.join(dir_path, 'source')
sys.path.append(source_path)
from models import UNet, NestedUNet
from models.VanillaCNN import VanillaCNN

# Set verbosity for external library
set_verbosity(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['NUNet', 'UNet', 'VanillaCNN'], default='UNet', help='NUNet, UNet, VanillaCNN')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-n', '--num_samples', type=int, default=1024, help='Number of samples')
    parser.add_argument('-s', '--data_path', type=str, default=os.path.join(dir_path, 'data', 'training_data'), help='Path to data')
    parser.add_argument('-f', '--filename', type=str, default='ellipses', help='Filename of data')
    parser.add_argument('-g', '--generate_non_attenuated_sensitivity', action='store_true', help='include non-attenuated sensitivity')
    parser.add_argument('-r', '--train_valid_ratio', type=float, default=0.8, help='ratio of training to validation data')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--save_path', type=str, default=os.path.join(dir_path, 'data', 'trained_models'), help='path to save data')
    parser.add_argument('--scheduler', type=str, choices=['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'None'], default="None", help='scheduler type')
    parser.add_argument('--pretrained', type=str, help='path to pretrained model')
    parser.add_argument('--save_name', type=str, default='model', help='name of saved data')
    parser.add_argument('--data_suffix', type=int, default=0, help='suffix of saved data')
    return parser.parse_args()

def load_data(data_path, save_name, n_samples, train_valid_ratio, incl, suffix):
    if incl: 
        gen_attn = "plus_non_attenuated"
    else:
        gen_attn = "original"
    X = torch.load(os.path.join(data_path, gen_attn, f'{save_name}_X_train_n{n_samples}_{suffix}.pt'))
    y = torch.load(os.path.join(data_path, gen_attn, f'{save_name}_y_train_n{n_samples}_{suffix}.pt'))
    train_size = int(train_valid_ratio * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_valid, y_valid = X[train_size:], y[train_size:]
    
    # Ensure that data with 1 channel is expanded to a 4D tensor
    if X_train.ndim == 3:
        X_train = X_train.unsqueeze(1)
        X_valid = X_valid.unsqueeze(1)
    if y_train.ndim == 3:
        y_train = y_train.unsqueeze(1)
        y_valid = y_valid.unsqueeze(1)
    
    return X_train, y_train, X_valid, y_valid

def get_model(model_name, in_channels=3, out_channels=1):
    models = {
        'NUNet': NestedUNet(in_channels, out_channels),
        'UNet': UNet(in_channels, out_channels),
        'VanillaCNN': VanillaCNN(in_channels, out_channels)
    }
    if model_name in models:
        return models[model_name]
    else:
        raise ValueError(f'Unknown model name {model_name}')

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

def plot_loss(train_losses, valid_losses, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters for grid search
    learning_rates = [0.001]
    batch_sizes = [16, 32]
    epochs_list = [100, 200, 500, 1000]
    models = ['VanillaCNN']

    results = []

    for lr, batch_size, n_epochs, model_name in product(learning_rates, batch_sizes, epochs_list, models):
        print(f"\nTraining {model_name} with lr={lr}, batch_size={batch_size}, epochs={n_epochs}")

        # Load data
        X_train, y_train, X_valid, y_valid = load_data(
            args.data_path, args.filename, args.num_samples,
            args.train_valid_ratio, args.generate_non_attenuated_sensitivity, args.data_suffix
        )

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False)

        in_channels = 3 if args.generate_non_attenuated_sensitivity else 2
        out_channels = 1
        model = get_model(model_name, in_channels, out_channels).to(device)

        loss_fn = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Create scheduler if specified
        scheduler = None
        if args.scheduler == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        elif args.scheduler == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        elif args.scheduler == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.00001)

        train_losses, valid_losses = [], []
        save_dir = os.path.join(args.save_path, f"grid_{model_name}_lr{lr}_bs{batch_size}_e{n_epochs}")
        os.makedirs(save_dir, exist_ok=True)
        
        best_valid_loss = float('inf')
        best_epoch = -1

        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
            valid_loss = validate_epoch(model, valid_loader, loss_fn, device)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

            print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch +1
                # save best model
                save_checkpoint(model, optimizer, os.path.join(save_dir, 'best_model.pth'))


        # Save results
        results.append({
            'model': model_name,
            'learning_rate': lr,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'best_valid_loss': best_valid_loss,
            'best_epoch': best_epoch
        })

        

        save_checkpoint(model, optimizer, os.path.join(save_dir, 'final_model.pth'))
        plot_loss(train_losses, valid_losses, os.path.join(save_dir, 'loss_plot.png'))

        print(f"Saved model and loss plot to {save_dir}")

        # Copy to cratch output after each run
        scratch_output_path = Path("/home/ucapas9/Scratch/output/trained_models")
        scratch_output_path.mkdir(parents=True, exist_ok=True)
        try:
            os.system(f'cp -r {save_dir} {scratch_output_path}')
            print(f"Copied to scratch: {scratch_output_path}")
        except Exception as e:
            print(f"Copy failed: {e}")

    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(args.save_path, f"grid_search_results_{model_name}_lr{lr}.csv")
    results_df.to_csv(results_csv_path, index=False)



if __name__ == '__main__':
    main()
