"""
This is the main script for training and evaluating the models on the PneumoniaMNIST dataset.

Input:
    - model: str, name of the model to train
    - logdir: str, directory to save logs
    - batch_size: int, batch size
    - num_epochs: int, number of epochs
    - lr: float, learning rate
    - weight_decay: float, weight decay
    - print_every: int, print every

Author:
    Credits to the work done previously by Johan Obando and Jerry Huang
    Pierre-Louis Benveniste
"""
import argparse
import torch
from torch import optim
from torchvision import transforms
import json
from mlp import MLP
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import os
from medmnist import PathMNIST
import matplotlib.pyplot as plt
import numpy as np
from mobileNet import MobileNet


# seed experiment
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True


def parse_args():
    """
    This function parses the command line arguments.
    """
    parser = argparse.ArgumentParser(description='PneumoniaMNIST training script')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'mobilenet'], help='Model to train (default: %(default)s).')
    parser.add_argument('--logdir', type=str, default=None, help='Directory to save logs', required=True)
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: %(default)s).')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training (default: %(default)s).')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: %(default)s).')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: %(default)s).')
    parser.add_argument('--print_every', type=int, default=80, help='Number of minibatches after which we print the loss (default: %(default)s).')
    return parser.parse_args()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ Compute the accuracy of the batch """
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc


def train(epoch, model, dataloader, optimizer, loss_fn, accuracy_fn, device, args):
    model.train()
    total_iters = 0
    epoch_accuracy=0
    epoch_loss=0
    start_time = time.time()
    for idx, (X,y) in enumerate(dataloader):
        # Format the data
        y = y.squeeze().long()
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
 
        # Compute loss and accuracy
        batch_loss = loss_fn(y_pred, y)
        loss = batch_loss/args.batch_size
        acc = accuracy_fn(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update epoch accuracy and loss
        epoch_accuracy += acc.item() / len(dataloader)
        epoch_loss += loss.item() / len(dataloader)
        total_iters += 1

        # Print every args.print_every iterations
        if idx % args.print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")
    tqdm.write(f"== [TRAIN] Epoch: {epoch}, Accuracy: {epoch_accuracy:.3f} ==>")
    return epoch_loss, epoch_accuracy, time.time() - start_time


def evaluate(epoch, model, dataloader, loss_fn, accuracy_fn, device, args, mode="val"):
    model.eval()
    epoch_accuracy=0
    epoch_loss=0
    total_iters = 0
    start_time = time.time()
    with torch.no_grad():
        for idx, (X,y) in enumerate(dataloader):
            # Format the data
            y = y.squeeze().long()
            X, y = X.to(device), y.to(device)

            # Forward pass
            logits = model(X)
            
            # Compute loss and accuracy
            batch_loss = loss_fn(logits, y)
            loss = batch_loss/args.batch_size
            acc = accuracy_fn(logits, y)

            # Update epoch accuracy and loss
            epoch_accuracy += acc.item() / len(dataloader)
            epoch_loss += loss.item() / len(dataloader)
            total_iters += 1

            # Print every args.print_every iterations
            if idx % args.print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}"
                )
        tqdm.write(
            f"=== [{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Accuracy: {epoch_accuracy:.3f} ===>"
        )
    return epoch_loss, epoch_accuracy, time.time() - start_time


def main():
    # Parse the arguments
    args = parse_args()

    # Check for the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # We define the transformations used during training and validation
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(norm_mean, norm_std)
                                     ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((28,28),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)
                                        ])
    
    # Loading the training dataset. We need to split it into a training and validation part 
    train_set = PathMNIST(split='train', transform=train_transform, download=True)
    val_set = PathMNIST(split='val', transform=test_transform, download=True)
    test_set = PathMNIST(split='test', transform=test_transform, download=True)

    # Load model
    print(f'Build model {args.model.upper()}...')
    if args.model == 'mlp':
        model = MLP(input_size=____, hidden_sizes=[1024,512,64,64], num_classes=____, activation="relu")
    if args.model == 'mobilenet':
        model = MobileNet(num_classes=____)
    model.to(device)
    print(f"Initialized {args.model.upper()} model with {sum(p.numel() for p in model.parameters())} "
          f"total parameters, of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are learnable.")
    print("Model architecture:\n", model)
    print("\n")
    
    # Optimizer
    optimizer = optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Accuracy function
    accuracy_fn = compute_accuracy
    
    # Initialize lists to store metrics
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    train_times, valid_times = [], []
    
    # We define a set of data loaders that we can use for various purposes later.
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Train and evaluate the model
    print(f'Training {args.model.upper()} model...')
    for epoch in tqdm(range(args.epochs)):
        tqdm.write(f"====== Epoch {epoch} ======>")

        # Train the model
        loss, acc, wall_time = train(epoch, model, train_dataloader, optimizer, loss_fn, accuracy_fn, device, args)
        train_losses.append(loss)
        train_accs.append(acc)
        train_times.append(wall_time)

        # Evaluate the model
        loss, acc, wall_time = evaluate(epoch, model, valid_dataloader, loss_fn, accuracy_fn, device, args)
        valid_losses.append(loss)
        valid_accs.append(acc)
        valid_times.append(wall_time)

    # Test the model
    test_loss, test_acc, test_time = evaluate(epoch, model, test_dataloader, loss_fn, accuracy_fn, device, args, mode="test")
    print(f"===== Best validation Accuracy: {max(valid_accs):.3f} =====>")

    # Save log if logdir provided
    if args.logdir is not None:
        print(f'Writing training logs to {args.logdir}...')
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir, 'results.json'), 'w') as f:
            f.write(json.dumps(
                {
                    "train_losses": train_losses,
                    "valid_losses": valid_losses,
                    "train_accs": train_accs,
                    "valid_accs": valid_accs,
                    "test_loss": test_loss,
                    "test_acc": test_acc
                },
                indent=4,
            ))


if __name__ == "__main__":
    main()