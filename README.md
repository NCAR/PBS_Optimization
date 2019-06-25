# PBS Scheduler Optimization

This repository implements data preprocessing, machine learning models to optimize PBS Scheduler


# Requirements

Python 3.6.2 <br />
Numpy <br />
Pandas <br />
pytorch 1.0.1 <br />
scikit-learn 0.21.1 <br />
tensorflow 1.13.1 <br />

# Usage

```
python3 pbs_parser_seaprate_v3.py accounting
```
 where
	accounting is the location of directory with accounting logs (previous historical PBS data)
If you find the code useful, please cite the paper.


To extract additional features from accounting logs, please follow the commands

```
python3 process_data.py
```

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

# Training State-of-the-art Model (Neural Network)
To train either FeedForward Network (keyword: ff), Recurrent Neural Network (keyword: rnn), Convolutional Neural Network (keyword: cnn), or Residual Neural Network (keyword: resnet), please follow the example below. Example is for FeedForward network:

```
python3 train.py --batch_size=32 --num_epochs=200 --hidden_size=128 --ckpt=False --train_path='../training_small/' --ckpt_path='../best_ff_model/' --test_path='../testing_small/' --model_type='ff' --dropout=0.8 --device='cuda:0'
```
# Training State-of-the-art Model (Random Forest, XGBoost)
Training default model proposed by "Machine Learning Predictions for Underestimation of Job Runtime on HPC System" (Guo, Nomura, Barton, Zhang, and Matsuoka, 2018)
```
python3 rf_xgboost.py 
``
