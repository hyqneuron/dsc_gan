# Deep Subspace Clustering + GAN

Based on:

Pan Ji*, Tong Zhang*, Hongdong Li, Mathieu Salzmann, Ian Reid. Deep Subspace Clustering Networks. in NIPS'17.

Forked off: https://github.com/panji1990/Deep-subspace-clustering-networks

How to run:
```
CUDA_VISIBLE_DEVICES=0 python dsc_gan.py exp_name --lambda3 1 --epochs 3000
```

Dependencies:
```
pip install tensorflow-gpu==1.1
pip install tensorboard
```

tensorboard useful for visualization only.


# Notes on parameter tuning

Training result seems to be highly dependent on quality of pre-trained model. For a particular pretrained model,
finetuning achieves very consistent result. But if we repeat pre-training everytime, final finetuning result
has significant variance.


# TODO

[ ] Add COIL10 and COIL100
[ ] Allow tuning of number of conv-layers and disc size
[ ] Nxr rxN

