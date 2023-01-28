# Multi-Label Classification through Single_objective and Multi-Objective Optimizations

## Getting Started
This GitHub Repository contains implementation of two multi-label classification methods namely [Zhang's Convex Calibrated Surrogates for Multi-Label F-Measure](https://arxiv.org/abs/2009.07801#:~:text=The%20F%2Dmeasure%20is%20a,be%20active%20in%20any%20image) and [Sener's Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650). These methods are compared against the baseline of Binary Relevance. Both of these method handle the multi-label classification in different ways. Zhang's paper proposed a single surrogate-risk minimzation framework that can perform almost as good as the Bayes-optimal classifier for ***F<sub>β</sub>*** measure. However the computation complexity of the algorithm is high. On the other hand, Sener's paper viewed the MLC problem as Multi-task learning problem and proposed a learning algorithm that tries to minimize multiple surrogate-risk together (one for each label). Inshort, Sener's algorithm handles the interlabel dependencies by considering one classifier for each label and then training all of them together through multi-objective optimization. This is in contrast to the Binary relevance method, where the classifier for each label is trained separately. An interesting feature of SA is that it offers a much more flexible (tunable) computational complexity compared to ZA. Given this attractive feature, we wish to find out whether Sener's can be used as a reliable heuristic classifier for the ***F<sub>β</sub>***-measure. 
**Note** that this implementation and analysis has been done as part of the EECS-545 (Advanced Machine Learning) course project. 

## Setup
Dependencies :-
- torch
- torchvision
- Numpy
- Pandas
- sklearn
- tqdm

## File Structure
This repository contains Datasets, Final trained models for each algorithm, implementation scripts of each algorithm and testing scripts. 

1. **Dataset** directory contains 2 synthetic data (data1 and data2) and an emotion data (data3)
	- *./Data{1,2,3}* directory contains training, validation and testing numpy files for data1, data2, and data3 respectively. 
	- *./Data{1,2,3}/Metrics* directory contains evaluation metrics tested on testing data with the final trained model on each dataset.
2. **FinalModels** directory contains the best model for each algorithm and for each datasets.
3. **BR_1** directory contains Binary Relevance implementation. 
4. **Zhang_2** directory contains Zhang's algorithm implementation.
5. **Sener_3** directory contains Sener's algorithm implementation.
6. *scatterplot_data{1,2,3}.m* is a MATLAB script for data1, data2 and data3 respectively, which plots the learning curve i.e. "Number of training sample Vs Fbeta accuracy" for three different learning rates. 
7. *testallalgs_data{1,2,3}.py* is the testing script which takes the best model from *FinalModels* directory and evaluate the model on testing data for all the three algorithms.

## Execution
The README files in each directory provide instructions on how to execute the algorithm and the necessary packages for training and testing.

## Report
We compiled our finding into the *EECS_545_Group_Project_Report.pdf* report. The results and analysis of the implemented methods are also reported in the same document.

	
