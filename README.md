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

## Rate Shift Evaluation via DomainNet

DomainNet consists of six domains (e.g., ketch, clipart, painting, quickdraw, real, and infograph), each of which consists of images from 345 classes. 
We consider the all domains as the source and each domain, including all domains, as a target. 
Thus, we will have 7 adapted models as score functions of PAC prediction sets. 


### Dataset Initialization
First of all, we need to donwload and postprocess the [DomainNet](http://ai.bu.edu/M3SDA/) dataset; 
the following script downloads the entire DomainNet datasets and holds out a validation set from a test set for you.
```
./scripts/init_domainnet_dataset.sh
```
### Model Initilization
For each shift, we train a domain adapted classifier via [DANN](https://arxiv.org/pdf/1505.07818.pdf). 
The trained models are provided in this repository and initialized via the following script. 
```
./scripts/init_models.sh
```

### Learning

### Calibration
```
./demo.sh
```

## Citation

```
@inproceedings{
    park2022pac,
    title={{PAC} Prediction Sets Under Covariate Shift},
    author={Sangdon Park and Edgar Dobriban and Insup Lee and Osbert Bastani},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=DhP9L8vIyLc}
}
```
