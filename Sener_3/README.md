## File structure.
This folder contains implementation scripts for Sener's (SA) algorithm on mainly three dataset which includes 2 synthetic dataset(data1 and data3) and an emotion dataset(data2). The script structure are as follows:-

1. ``sener_data1.py``, ``sener_data2.py`` and ``sener_data3.py`` are the main training script file for SA for data1, data and data3 respectively by taking the shallowest network (SA-1).
2. ``sener_data1_dnn2.py``, ``sener_data2_dnn2.py`` and ``sener_data3_dnn2.py`` are the main training script file for SA for data1, data and data3 respectively by taking the considerably deep network (SA-2).
3. ``sener_data1_dnn3.py``, ``sener_data2_dnn3.py`` and ``sener_data3_dnn3.py`` are the main training script file for SA for data1, data and data3 respectively by taking the deepest network (SA-3).
4. ``model_loader1.py``, ``model_loader2.py`` and ``model_loader3.py`` is a PyTorch model architecture file for SA-1.
5. ``model_loader1_dnn2.py``, ``model_loader2_dnn2.py`` and ``model_loader3_dnn2.py`` is a PyTorch model architecture file for SA-2.
6. ``model_loader1_dnn3.py``, ``model_loader2_dnn3.py`` and ``model_loader3_dnn3.py`` is a PyTorch model architecture file for SA-3.
5. ``loaders.py`` is a DataLoader file to import data in batches for data1 and data3. ``loaders2.py`` is for data2.
6. ``losses.py`` contains the loss function. 

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
- Run ``sener_data1.py`` file to train the SA on data1 for SA-1, run ``sener_data2.py`` file to train the SA on data2 for SA-1, and run the ``sener_data3.py`` for data3 for SA-1. 
- Run ``sener_data1_dnn2.py`` file to train the SA on data1 for SA-2, run ``sener_data2_dnn2.py`` file to train the SA on data2 for SA-2, and run the ``sener_data3_dnn2.py`` for data3 for SA-2. 
- Run ``sener_data1_dnn3.py`` file to train the SA on data1 for SA-3, run ``sener_data2_dnn3.py`` file to train the SA on data2 for SA-2, and run the ``sener_data3_dnn3.py`` for data3 for SA-3. 

Each of the script save training model after every 9th epoch inside *./Data{1,2,3}Models* depending upon the script and dataset chosen. 


