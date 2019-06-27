# PBS Scheduler Optimization

This repository implements data preprocessing, machine learning models to optimize PBS Scheduler


# Requirements

Python 3.6.2 <br />
Numpy <br />
Pandas <br />
Pytorch 1.0.1 <br />
Scikit-learn 0.21.1 <br />
Tensorflow 1.13.1 <br />

# Data Preprocessing

```python
python3 pbs_parser_seaprate_v3.py accounting
```
 where
	accounting is the location of directory with accounting logs (previous historical PBS data)
If you find the code useful, please cite the paper.


To extract additional features from accounting logs, please follow the commands

```python
python3 process_data.py
```

# Data Analysis & Plotting
Current list of supported plotting features:
	Resource_List.walltime <br />
	Resource_List.ncpus <br />
	Resource_List.mpiprocs <br />
	Resource_List.nodect <br />
	resources_used.walltime <br />
	resources_used.cput <br />
	user_mispred <br />
	mispred_ratio <br />
	mispred_ratio_walltime <br />
	
Current list of supported groupby_val features:
	user <br />
	account <br />
---
To generate single feature plots, follow the commands
```
python3 create_feature_plots.py ABC
```
where ABC stands for the features (i.e. queue, resources_used.walltime, etc.)

To generate multi-feature plots, follow the commands
 ```
python3 create_multi-index_feature_plots.py ABCandDEF
```
where ABC and DEF stand for the two different features (i.e. queue, resources_used.walltime, etc.)

To generate 2-dimensional plot analysis (by 2 features), follow the commands:
```python
python3 overall_plot.py --multi_dim_plot=True --multi_dim_x_feature= mispred_ratio, --multi_dim_y_feature='num_jobs' --num_top=10 --groupby_val='account' --data_path ='../apr_2019_full/'
```
Argument parameters:
 --multi_dim_plot: whether to plot multi_dim or not, default: False
 --multi_dim_x_feature: features on x_axis, default: mispred_ratio
 --multi_dim_y_feature: features on y_axism defaultL num_jobs
 --num_top: top users by two 2 specified dimensions (multiplication), default: 0 (no filter)
 --groupby_val: data point representation (i.e. user/ account), default: user
 --data_path: location storing parsed accounting csv of interest

To generate overall_plot, follow the commands:
```python
python3 overall_plot.py --data_path='../apr_2019_full/' --groupby_val='user' --overall_distr_plot=True --overall_feature='user_mispred'
```
Argument parameters:
  --data_path: location storing parsed accounting csv of interest
  --groupby_val: data point representation (i.e. user/ account), default: user
  --overall_distr_plot: whether to plot overall distribution or not
  --overall_feature: feature to plot by

# Training State-of-the-art Model (Neural Network)
To train either FeedForward Network (keyword: ff), Recurrent Neural Network (keyword: rnn), Convolutional Neural Network (keyword: cnn), or Residual Neural Network (keyword: resnet), please follow the example below. Example is for FeedForward network:

```python
python3 train.py --batch_size=32 --num_epochs=200 --hidden_size=128 --ckpt=False --train_path='../training_small/' --ckpt_path='../best_ff_model/' --test_path='../testing_small/' --model_type='ff' --dropout=0.8 --device='cuda:0'
```
# Training State-of-the-art Model (Random Forest, XGBoost)
Training default model proposed by "Machine Learning Predictions for Underestimation of Job Runtime on HPC System" (Guo, Nomura, Barton, Zhang, and Matsuoka, 2018)

```python
python3 rf_xgboost.py 
```
# Unsupervised Domain Adaptation by Backpropagation
Implementation of the proposed domain adaptation model from "Unsupervised Domain Adaptation by Backpropagation" (Ganin, Lempitsky, 2015)

```python
python3 train_transfer.py --batch_size=32 --num_epochs=200 --hidden_size=128 --ckpt=False --train_path='../training_small/' --ckpt_path='../best_dann_model/' --test_path='../testing_small/' --model_type='dann' --dropout=0.8 --device='cuda:0'
```

# Deep Adaptation Network (DAN)
Implementation of the proposed domain adaptation model from "Learning Transferable Features with Deep Adaptation Networks" (Long, Cao, Wang, Jordan, 2015)

```python
python3 train_transfer.py --batch_size=32 --num_epochs=200 --hidden_size=128 --ckpt=False --train_path='../training_small/' --ckpt_path='../best_dan_model/' --test_path='../testing_small/' --model_type='dan' --dropout=0.8 --device='cuda:0'
```
```

