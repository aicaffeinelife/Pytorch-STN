# Pytorch-STN
Spatial Transformer Networks in Pytorch.


This repository contains a PyTorch implementation of [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) by Jaderberg
et al. The results are reported on the CIFAR-10 dataset and SVHN results will be coming up shortly. 



## Training your own model 

Training is made to be very simple. You define your own experiment directory under `experiments` folder and populate it with a 
`params.json`. Example configs can be found in `experiments\base_svhn` and `experiments\stn_svhn`. To train a model in `models`
folder:

```
python train.py --param_path <path_to_experiment> --resume_path <last checkpoint> 
```

The code frequently stores the checkpoints as `last.pth.tar` corresponding to the last epoch run and `best.pth.tar` corresponding 
to the checkpoint which had the best validation set accuracy. Training can be resumed by giving that parameter to the train script. 

For example if you want to train a base Network and want to fine tune from the last best checkpoint you can write:

```
python train.py -- param_path experiments/base_svhn --resume_path best
```

## Results 


|Dataset| Model|Hardware | Epochs | Validation Accuracy | 
|-----| :-------| :--------| :-------|:-------|
|CIFAR-10 | Base | Gtx-1080 | 150 | 70.9% |
|CIFAR-10 | STN-Net| Gtx-1080 | 150 | 76.96% |
