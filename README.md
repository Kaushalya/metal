# Active Graph Learning

**Pytorch** implementation of the paper ["MetAL: Active Semi-Supervised Learning on Graphs via Meta Learning"](http://proceedings.mlr.press/v129/madhawa20a.html) (2020)

## Dependencies
1. python 3.6+
1. pytorch 1.4+
1. numpy
1. scipy
1. networkx
1. scikit-learn
1. timebudget

## Run the code

Make sure you have installed all requirements.

Run an example experiment with: 
`sh run_active_learn.sh`

This will run 2 trials of meta active learning on CiteSeer
dataset and save the performance to a csv file.

## Output
Once you execute an active learning experiment, performance is
saved as a csv file in the `results` directory.
For example, running 10 trials of entropy acquisition function will
create a csv file with the name `citeseer_entropy_10trials-accuracy.csv`.

### Structure of the results file
Each row corresponds to an acquisition of a set of nodes. For each 
acquisition accuracy, macro-f1, and micro-f1 of the test set is saved
along columns of the csv file.

## Visualization of results
The Jupyter notebook `notebooks/analysis/performance_summary.ipynb` is used to
load results CSV files and to plot how the performance (accuracy and macro-f1) varies
with acquisition of labels of the unlabeled nodes.

## Cite
Please cite our paper if you use this code in your own work:
```
 @InProceedings{madhawa20metal,
title = {{M}et{A}{L}: {A}ctive {S}emi-{S}upervised {L}earning on {G}raphs via {M}eta-{L}earning},
author = {Madhawa, Kaushalya and Murata, Tsuyoshi}, 
booktitle = {Proceedings of The 12th Asian Conference on Machine Learning}, 
pages = {561--576}, 
year = {2020}, 
editor = {Sinno Jialin Pan and Masashi Sugiyama}, 
volume = {129}, 
series = {Proceedings of Machine Learning Research}, 
address = {Bangkok, Thailand}, month = {18--20 Nov}, 
publisher = {PMLR}, 
pdf = {http://proceedings.mlr.press/v129/madhawa20a/madhawa20a.pdf}, 
url = {http://proceedings.mlr.press/v129/madhawa20a.html}} 
```
