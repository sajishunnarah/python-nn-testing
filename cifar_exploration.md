# CIFAR-10 Exploration

## This is my journey of exploring the CIFAR-10 dataset with different CNNs

# Process

I first made a generic trainer and tester to train and evaluate different CNN models on the data set and visualize the results over time.
Initially, I had a 2 layer CNN model and tuned hyperparameters to get around a 70% validation accuracy.
I then tried to make it a 3 layer CNN model, using a padding of two to keep higher dimensions to fit in another conv2d layer. The model got up to around 72% early. It got up to around 73.8% with more epochs but with around 80% training accuracy, indicating overfitting.

# Device testing
For the cifar_regular, I got the following: 
- 7 min 41 seconds for CPU
- 5 min 45 seconds for MPS
- 2 min X seconds for GPU on CoLab (didnt specify minutes)