### File structure.

This folder contains implementation scripts for Sener algorithm on MultiMNIST data. The script structure are as follows:-
1. "MultiObjectiveMinimization.py" is the main training and testing script file. 
2. "loaders.py" creates a PyTorch data loader.
3. "multi_mnist_loader.py" is a utils script for creating multi label dataset from MNIST data. Taken directly from SA author code with few changes.
4. "model_loader.py" is a PyTorch model architecture file.
5. "losses.py" contains loss function information for each tasks.

### Required packages.
The code uses the following Python packages and they are required: 
- tensorboardX
- pytorch
- click
- numpy
- torchvision
- tqdm
- scipy
- Pillow

### Training the model.
Run "MultiObjectiveMinimization.py" file, It will save training model after every 3 epoch inside "./saved_model" directory. After training it will show run on testing data and give you accuracy on each task.

 
