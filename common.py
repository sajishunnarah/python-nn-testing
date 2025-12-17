import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

## Switching to a configurable NN trainer. This can be resused.
def train(model, data, test_set, batch_size=64, weight_decay=0.0,
          optimizer="sgd", learning_rate=0.1, momentum=0.9,
          data_shuffle=True, num_epochs=4):
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
        train_acc_hist.append(get_accuracy(model, test_set, train=True))
        val_acc_hist.append(get_accuracy(model, test_set, train=False))
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