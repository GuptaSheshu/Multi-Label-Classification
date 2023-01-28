## File structure.
This folder contains implementation scripts for Binary Relevance(BR) algorithm on mainly three dataset which includes 2 synthetic dataset(data1 and data3) and an emotion dataset(data2). The script structure are as follows:-
1. ``br_data1.py`` is the main training script file for BR for data1.
2. ``br_data2.py`` is the main training script file for BR for data2.
3. ``br_data3.py`` is the main training script file for BR for data3.
4. ``model_loader.py`` is a PyTorch model architecture file.

## Packages.

- pytorch
- numpy
- torchvision
- scipy
- Pillow
- Pandas
- tqdm
- Sklearn


## Training the model.
- run ``br_data1.py`` file to train the BR on data1.
- run ``br_data2.py`` file to train the BR on data2
- run ``br_data3.py`` file to train the BR on data3. 

The script save training model after every 9th epoch inside *./Data{1,2,3}Models* depending upon the script and dataset chosen. 


