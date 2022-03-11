# PAC Prediction Sets Under Covariate Shift

This repository is the PyTorch implementation of 
[PAC Prediction Sets Under Covariate Shift](https://openreview.net/pdf?id=DhP9L8vIyLc) (ICLR22). 
A prediction set is a set-valued function that returns a subset of labels, where the set size represents the uncertainty of the prediction. 
A PAC prediction set is a prediction set that satisfies the probably approximately correct (PAC) guarantee, introduced by Wilks (1941) and Valiant (1984). 

The PAC prediction set is a promising way to measure the uncertainty of predictions, while making correctness guarantee, but 
it relies on the i.i.d. assumption, i.e., labeled examples are independent and identically distributed, which is easily broken in practice due to covariate shift. 

We propose a PAC prediction set algorithm that satisfies the PAC guarantee under covariate shift, while minimizing the expected set size; 
the following teaser summarizes our results.

![](.github/teaser.png)

## DomainNet Experiments

### Dataset Initialization
```
ln -s <path-to-domainnet-dataset> data/domainnet
```

### Learning

### Calibration
```
./demo.sh
```
