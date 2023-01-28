import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
import losses
import loaders
from model_loader import get_model
from mgda import MGDA_UB


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define HyperParameters
epochs = 100
lr = 0.01
batchsize = 256
tasks = ['L', 'R'];

# Load Dataset
data_path = './MULTIMNIST_DATASET'

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader, train_dst, val_loader, val_dst = loaders.get_dataset(dataset_name = "Multi_MNIST", transforms = transformation, dataset_path = data_path, batchsize=batchsize,download=True)


model = get_model(dataset_name="Multi_MNIST",parallel=False);
model_params = []
for m in model:
    model_params += model[m].parameters()


# Define loss and optimizer
nll_loss = losses.get_loss(tasks)
optimizer = torch.optim.SGD(model_params, lr=lr, momentum=0.9)


# Accurcy function
def get_accuracy(labels,ypred,ml,mr,t):
	ypred = np.argmax(ypred.cpu().detach().numpy(),1);
	if(t=='L'):
		ml.append(np.sum(ypred==labels['L'].detach().numpy()))
	elif(t=='R'):
		mr.append(np.sum(ypred==labels['R'].detach().numpy()))
	return ml,mr;

# Training Loop
print("===================================Start Training===================================")

n_iter = 0
loss_init = {}
ml_t,mr_t=[],[];
ml_v,mr_v=[],[];

for epoch in tqdm(range(epochs)):
	ml_t,mr_t=[],[];
	ml_v,mr_v=[],[];
	print('Epoch {} Started'.format(epoch))
	if (epoch+1) % 10 == 0:
	    # Every 50 epoch, half the LR
	    for param_group in optimizer.param_groups:
	        param_group['lr'] *= 0.85
	    print('Half the learning rate{}'.format(n_iter))

	for m in model:
	    model[m].train()

	for batch in train_loader:
	    n_iter += 1
	    X_train_batch = batch[0]
	    labels = {}
	    for i, t in enumerate(tasks):
	        labels[t] = batch[i+1]
	
	    # Scaling the loss functions based on the algorithm choice
	    loss_data = {}
	    grads = {}
	    scale = {}
	    mask = None
	    masks = {}

	    # MGDA-UB loop
	    optimizer.zero_grad()
	    X_train_batch = X_train_batch.data;
	    rep = model['rep'](X_train_batch)
	    
	    # Gradient wrt z(latent variable)
	    rep_variable = Variable(rep.data.clone(), requires_grad=True)
	    for t in tasks:
	    	optimizer.zero_grad()
	    	y_pred = model[t](rep_variable)
	    	loss = nll_loss[t](y_pred, labels[t])
	    	loss_data[t] = loss.data
	    	loss.backward()
	    	grads[t] = []
	    	grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
	    	rep_variable.grad.data.zero_()
	    # Frank-Wolfe solver
	    sol, min_norm = MGDA_UB.FW_solver([grads[t][0] for t in tasks])
	    for i, t in enumerate(tasks):
	    	scale[t] = float(sol[i])
	    # Scaled back-propagation
	    optimizer.zero_grad()
	    rep = model['rep'](X_train_batch)
	    for i, t in enumerate(tasks):
	    	y_pred_t = model[t](rep)
	    	loss_t = nll_loss[t](y_pred_t, labels[t])
	    	ml_t,mr_t = get_accuracy(labels,y_pred_t,ml_t,mr_t,t);
	    	loss_data[t] = loss_t.data
	    	if i > 0:
	    		loss = loss + scale[t]*loss_t
    		else:
    			loss = scale[t]*loss_t
	    loss.backward()
	    optimizer.step()

	## Evaluating model
	for m in model:
	    model[m].eval()

	tot_loss = {}
	tot_loss['all'] = 0.0
	met = {}
	for t in tasks:
	    tot_loss[t] = 0.0
	    met[t] = 0.0

	num_val_batches = 0
	for batch_val in val_loader:
	    X_val_batch = batch_val[0];
	    labels_val = {}

	    for i, t in enumerate(tasks):
	        labels_val[t] = batch_val[i+1]
	        
	    val_rep= model['rep'](X_val_batch)
	    for t in tasks:
	        y_pred_t_val = model[t](val_rep)
	        loss_t = nll_loss[t](y_pred_t_val, labels_val[t])
	        tot_loss['all'] += loss_t.data
	        tot_loss[t] += loss_t.data
	        ml_v,mr_v = get_accuracy(labels_val,y_pred_t_val,ml_v,mr_v,t);
	    num_val_batches+=1

	for t in tasks:
		print("Val_loss "+t+":",tot_loss[t]/num_val_batches);
		print("Val_loss "+t+":",loss_data[t]);
	print("Val_acc L:",np.sum(ml_v)/(100*100));
	print("Val_acc R:",np.sum(mr_v)/(100*100));
	print("Train_acc L:",np.sum(ml_t)/60000);
	print("Train_acc R:",np.sum(mr_t)/60000)

	if epoch % 3 == 0:
	    # Save after every 3 epoch
	    state = {'epoch': epoch+1,
	            'model_rep': model['rep'].state_dict(),
	            'optimizer_state' : optimizer.state_dict()}
	    for t in tasks:
	        key_name = 'model_{}'.format(t)
	        state[key_name] = model[t].state_dict()

	    torch.save(state, "saved_models/mgda_acc_LRLR_{}_{}_{}_{}_{}_model.pkl".format(epoch,np.sum(ml_t)/60000,np.sum(ml_v)/(100*100),np.sum(mr_v)/(100*100),np.sum(mr_t)/60000))

print("===================================Training Finished===================================")





## Testing script to verify the accuracy of final trained model.

misclass_left = []
misclass_right = []
for batch_val in val_loader:
    val_images = batch_val[0];
    labels_val = {}

    for i, t in enumerate(all_tasks):
        if t not in tasks:
            continue
        labels_val[t] = batch_val[i+1]
        labels_val[t] = labels_val[t];

    val_rep, _ = model['rep'](val_images, None)
    for t in tasks:
        out_t_val, _ = model[t](val_rep, None)
        ypred = np.argmax(out_t_val.detach().numpy(),1);
        if(t=='L'):
        	misclass_left.append(np.sum(ypred==labels_val[t].detach().numpy()))
        elif(t=='R'):
        	misclass_right.append(np.sum(ypred==labels_val[t].detach().numpy()))

left_acc = np.sum(misclass_left)/(100*100)
right_acc = np.sum(misclass_right)/(100*100)

