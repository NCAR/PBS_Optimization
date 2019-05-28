# PBS Scheduler Optimization

This repository implements data preprocessing, machine learning models to optimize PBS Scheduler


# Requirements

Python 3.6.2
Numpy
Pandas


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


