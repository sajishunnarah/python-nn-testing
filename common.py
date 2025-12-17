import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import ray
from ray import tune, train as ray_train
from ray.tune.schedulers import ASHAScheduler
import tempfile
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

## Switching to a configurable NN trainer. This can be resused.
def train(model, data, test_set, batch_size=64, weight_decay=0.0,
          optimizer="sgd", learning_rate=0.1, momentum=0.9,
          data_shuffle=True, num_epochs=4):
    print(f"Using device: {device}")
    model.to(device) # Move model to the selected device
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=batch_size,
                                               shuffle=data_shuffle,
                                               num_workers=2 # Added num_workers
                                               )
    # loss func
    criterion = nn.CrossEntropyLoss()
    # optimizer
    assert optimizer in ("sgd", "adam") # has to be one of deez
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)

    # track learning curve
    iters, losses, train_acc_hist, val_acc_hist = [], [], [], [] # Renamed for clarity

    # training
    n = 0 # iteration counter
    for epoch in range(num_epochs):
        running_loss = 0.0 # Track loss per epoch for printing
        for i, (imgs, labels) in enumerate(train_loader):
            if imgs.size()[0] < batch_size: # do we need this? what it do
                continue

            # Move inputs and labels to the selected device
            imgs, labels = imgs.to(device), labels.to(device)

            model.train()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save the info
            iters.append(n)
            losses.append(loss.item()) # average loss per img
            running_loss += loss.item()

            # Adjusted print frequency
            if (n + 1) % 500 == 0: # Print every 500 iterations
              print(f'[{epoch+1}, {i + 1:5d}] loss: {running_loss/500:.3f}') # Print average loss
              running_loss = 0.0
            n+=1

        # Calculate accuracy once per epoch
        train_acc_hist.append(get_accuracy(model, data, test_set, train=True))
        val_acc_hist.append(get_accuracy(model, data, test_set, train=False))
        print(f'Epoch {epoch+1} - Train Acc: {train_acc_hist[-1]:.4f}, Val Acc: {val_acc_hist[-1]:.4f}')

    # plottin
    plt.figure(figsize=(10, 5))
    plt.title("Learning Curve - Loss")
    plt.plot(iters, losses, label="Train Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Learning Curve - Accuracy")
    # Plot accuracy against epochs (as it's recorded once per epoch)
    plt.plot(range(1, num_epochs + 1), train_acc_hist, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), val_acc_hist, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()

    print(f"Final Training Accuracy: {train_acc_hist[-1]}")
    print(f"Final Validation Accuracy: {val_acc_hist[-1]}")

def get_accuracy(model, trainset, testset, train=False):
    model.to(device) # Ensure model is on the correct device for evaluation
    if train:
        data_loader = torch.utils.data.DataLoader(trainset, batch_size=4096, num_workers=2)
    else:
        data_loader = torch.utils.data.DataLoader(testset, batch_size=1024, num_workers=2)
    model.eval() # Annotate model for evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            # Move inputs and labels to the selected device
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            pred = output.max(1, keepdim=True)[1] # get index of the max log-probability?
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += imgs.shape[0]
    return correct/total


def tune_hyperparameters(model_class, trainset_ref, testset_ref, num_samples=10, max_num_epochs=10, gpus_per_trial=0):
    """
    Use Ray Tune to find optimal hyperparameters for your model.

    Args:
        model_class: The model class (not an instance, but the class itself, e.g., Net)
        trainset_ref: Ray object reference to training dataset (use ray.put(trainset))
        testset_ref: Ray object reference to test dataset (use ray.put(testset))
        num_samples: Number of hyperparameter configurations to try
        max_num_epochs: Maximum number of epochs per trial
        gpus_per_trial: Number of GPUs per trial (0 for CPU)

    Returns:
        Ray Tune ResultGrid with all trial results

    Example:
        import ray
        ray.init()
        trainset_ref = ray.put(trainset)
        testset_ref = ray.put(testset)
        results = common.tune_hyperparameters(Net, trainset_ref, testset_ref, num_samples=20)
        best = results.get_best_result("accuracy", "max")
        print(f"Best config: {best.config}")
    """
    def train_tune(config):
        """Training function called by Ray Tune for each trial"""
        # Get datasets from Ray object store
        trainset = ray.get(trainset_ref)
        testset = ray.get(testset_ref)

        # Initialize model
        model = model_class()
        model.to(device)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Optimizer based on config
        if config["optimizer"] == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=config["lr"],
                momentum=config.get("momentum", 0.9),
                weight_decay=config.get("weight_decay", 0.0)
            )
        else:  # adam
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["lr"],
                weight_decay=config.get("weight_decay", 0.0)
            )

        # Data loaders
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=2
        )

        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=1024,
            shuffle=False,
            num_workers=2
        )
        
        # Training loop
        for epoch in range(config["num_epochs"]):
            model.train()
            running_loss = 0.0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(test_loader)
            
            # Report to Ray Tune with checkpoint
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint_path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                ray_train.report(
                    {"loss": avg_val_loss, "accuracy": val_accuracy},
                    checkpoint=ray_train.Checkpoint.from_directory(temp_checkpoint_dir)
                )
    
    # Define hyperparameter search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "optimizer": tune.choice(["sgd", "adam"]),
        "momentum": tune.uniform(0.8, 0.99),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "num_epochs": max_num_epochs,
    }
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    
    # Run hyperparameter search
    tuner = tune.Tuner(
        tune.with_resources(
            train_tune,
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    
    results = tuner.fit()
    
    # Print best results
    best_result = results.get_best_result("accuracy", "max")
    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS FOUND:")
    print("="*70)
    print(f"Learning Rate:    {best_result.config['lr']:.6f}")
    print(f"Batch Size:       {best_result.config['batch_size']}")
    print(f"Optimizer:        {best_result.config['optimizer']}")
    print(f"Momentum:         {best_result.config['momentum']:.4f}")
    print(f"Weight Decay:     {best_result.config['weight_decay']:.6f}")
    print("-"*70)
    print(f"Best Accuracy:    {best_result.metrics['accuracy']:.4f}")
    print(f"Best Loss:        {best_result.metrics['loss']:.4f}")
    print("="*70 + "\n")
    
    return results