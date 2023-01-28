## File structure.
This folder contains implementation scripts for Zhang's (ZA) algorithm on mainly three dataset which includes 2 synthetic dataset(data1 and data3) and an emotion dataset(data2). The script structure are as follows:-
1. ``zhang_data1.py`` is the main training script file for BR for data1.
2. ``zhang_data2.py`` is the main training script file for BR for data2.
3. ``zhang_data3.py`` is the main training script file for BR for data3.
4. ``model_loader.py`` is a PyTorch model architecture file.
5. ``loaders.py`` is a DataLoader file to import data in batches.
6. ``generate_data1.py`` and ``generate_data2.py`` is script for generating the synthetic data mainly data1 and data2.
7. ``losses.py`` contains the zhang loss function. It is custom defined. 

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
- run ``zhang_data1.py`` file to train the ZA on data1
- run ``zhang_data2.py`` file to train the ZA on data2
- run ``zhang_data3.py`` file to train the ZA on data3. 
The script save training model after every 9th epoch inside *./Data{1,2,3}Models* depending upon the script and dataset chosen. 


