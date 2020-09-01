# Active Graph Learning

**Pytorch** implementation of the paper ["MetAL: Active Semi-Supervised Learning on Graphs via Meta Learning"](https://arxiv.org/abs/2007.11230) (2020)

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
@article{madhawa2020metal,
  title={MetAL: Active Semi-Supervised Learning on Graphs via Meta Learning},
  author={Madhawa, Kaushalya and Murata, Tsuyoshi},
  journal={arXiv preprint arXiv:2007.11230},
  year={2020}
}
```
