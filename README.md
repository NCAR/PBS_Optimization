# PBS Scheduler Optimization

This repository implements data preprocessing, machine learning models to optimize PBS Scheduler


# Requirements

Python 3.6.2 <br />
Numpy <br />
Pandas <br />


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
python3 create_feature plots.py ABC
```
where ABC stands for the features (i.e. queue, resources_used.walltime, etc.)

To generate multi-feautre plots, follow the commands
 ```
python3 create_feature plots.py ABCandDEF
```
where ABC and DEF stand for the two different features (i.e. queue, resources_used.walltime, etc.)

