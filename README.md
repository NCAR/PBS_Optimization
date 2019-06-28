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

```
python3 pbs_parser_seaprate_v3.py accounting
```
 where
	**accounting** is the location of directory with accounting logs (previous historical PBS data)

To extract additional features from accounting logs, follow the commands

```
python3 process_data.py
```
To split data ino different weeks, follow the commands
```
python3 weekly_split.py pre_data_loc post_data_loc
```
where **pre_data_loc** : directory containing pre-processed CSV data <br />
      **post_data_loc** : directory containing post-processed CSV data

# Data Analysis & Plotting
The goal of the scripts in this section is to produce charts for visualization of PBS accounting data. The analysis is done by calculating jobs' misprediction and dividing them into 6 bins (0-15 minutes, 15minutes-1hour, 1hour-3hours, 3hours-7hours, more than 7hours)

## Plot by single feature
This script produces breakdown of users' misprediction in terms of different user_selected features. Follow the command to execute the script
```
python3 create_feature_plots.py data_path feature
```
where
	**data_path** directory containing CSV data <br />
	**feature** stands for the features (i.e. queue, resources_used.walltime, etc.)

Successful scripts would produce 2 types of plots:
	1. Pie plot: for individual period dissected by feature (i.e. Monday-> Friday if feature is day_week) <br />
	2. Stacked column chart: for all of the periods dissected by feature 

## Plot by 2 features 
This script extends the previous script to produce two-level feature filters. Currently, feature2 is supposed to be an outer filter and feature1 is an inner filter
 ```
python3 create_multi-index_feature_plots.py feature1andfeature2
```

where **feature1** and **feature2** stand for the two different features (i.e. queue, resources_used.walltime, etc.)

Successful scripts would produce 2 types of plots:
	1. Pie plot: Contribution of each **feature2** with more than 2% of job volume <br />
	2. Stacked column chart: for each **feature2** with components of **feature1**

(See example in img directory)

**Current list of supported plotting features:** <br />
	time_day <br />
	week_month <br />
	day_week <br />
	user <br />
	account <br />
	queue <br />

## 2-dimensional plot (scatterplot)
To generate 2-dimensional plot analysis (by 2 features), follow the commands:
```
python3 overall_plot.py --multi_dim_plot=True --multi_dim_x_feature='mispred_ratio', --multi_dim_y_feature='num_jobs' --num_top=10 --groupby_val='account' --data_path='../apr_2019_full/'
```
Argument parameters:
 --**multi_dim_plot** whether to plot multi_dim or not, default: False <br />
 --**multi_dim_x_feature** features on x_axis, default: mispred_ratio <br />
 --**multi_dim_y_feature** features on y_axism defaultL num_jobs <br />
 --**num_top** top users by two 2 specified dimensions (multiplication), default: 0 (no filter) <br />
 --**groupby_val** data point representation (i.e. user/ account), default: user <br />
 --**data_path** location storing parsed accounting csv of interest <br />

## Overall plot distribution
To generate overall_plot, follow the commands:
```
python3 overall_plot.py --data_path='../apr_2019_full/' --groupby_val='user' --overall_distr_plot=True --overall_feature='user_mispred'
```
Argument parameters:
  --**data_path** location storing parsed accounting csv of interest <br />
  --**groupby_val** data point representation (i.e. user/ account), default: user <br />
  --**overall_distr_plot** whether to plot overall distribution or not <br />
  --**overall_feature** feature to plot by <br />

**Current list of supported plotting features:** <br />
	Resource_List.walltime <br />
	resources_used.walltime <br />
	resources_used.cput <br />
	user_mispred <br />
	mispred_ratio <br />
	mispred_ratio_walltime <br />
	
**Current list of supported groupby_val features:** <br />
	user <br />
	account <br />
# Training State-of-the-art Model (Neural Network)
To train either FeedForward Network (keyword: ff), Recurrent Neural Network (keyword: rnn), Convolutional Neural Network (keyword: cnn), or Residual Neural Network (keyword: resnet), please follow the example below. Example is for FeedForward network:

```
python3 train.py --batch_size=32 --num_epochs=200 --hidden_size=128 --ckpt=False --train_path='../training_small/' --ckpt_path='../best_ff_model/' --test_path='../testing_small/' --model_type='ff' --dropout=0.8 --device='cuda:0'
```
# Training State-of-the-art Model (Random Forest, XGBoost)
Training default model proposed by *"Machine Learning Predictions for Underestimation of Job Runtime on HPC System" (Guo, Nomura, Barton, Zhang, and Matsuoka, 2018)*

```
python3 rf_xgboost.py 
```
# Unsupervised Domain Adaptation by Backpropagation
Implementation of the proposed domain adaptation model from *"Unsupervised Domain Adaptation by Backpropagation" (Ganin, Lempitsky, 2015)*

```
python3 train_transfer.py --batch_size=32 --num_epochs=200 --hidden_size=128 --ckpt=False --train_path='../training_small/' --ckpt_path='../best_dann_model/' --test_path='../testing_small/' --model_type='dann' --dropout=0.8 --device='cuda:0'
```

# Deep Adaptation Network (DAN)
Implementation of the proposed domain adaptation model from *"Learning Transferable Features with Deep Adaptation Networks" (Long, Cao, Wang, Jordan, 2015)*

```
python3 train_transfer.py --batch_size=32 --num_epochs=200 --hidden_size=128 --ckpt=False --train_path='../training_small/' --ckpt_path='../best_dan_model/' --test_path='../testing_small/' --model_type='dan' --dropout=0.8 --device='cuda:0'
```
```

