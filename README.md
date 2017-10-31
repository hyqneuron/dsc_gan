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


# coil20 experiments

Without GAN

- coil20_1: pretrain 10000, epochs 1000, enableat 1000
- coil20_2: pretrain     1, epochs 1000, enableat 1000
- coil20_3: pretrain 10000, epochs 1000, enableat 1000, lambda4 100
- coil20_4: pretrain 10000, epochs 1000, enableat 1000, lambda1 100
- coil20_5: pretrain 10000, epochs 1000, enableat 1000, lambda1 10000
- coil20_6: pretrain 10000, epochs 1000, enableat 1000, lambda1 100000
- coil20_7: pretrain 10000, epochs 1000, enableat 1000, lambda1 1000000

- coil_m1: pretrain 3500, epochs 500, enableat1000, lambda2 20, alpha 0.12
- coil_m2: pretrain 3500, epochs 500, enableat1000, lambda2 20, alpha 0.12
- coil_m3: pretrain 3500, epochs 500, enableat1000, lambda2 20, alpha 0.12
- coil_m4: pretrain 3500, epochs 500, enableat1000, lambda2 20, alpha 0.12, lr0.0004
- coil_m5: pretrain 6000, epochs 700, enableat1000, lambda2 20, alpha 0.12, lr0.0004
- coil_m6: pretrain 6000, epochs 1000, enableat2000, lambda2 20, alpha 0.12, lr0.0004
- coil_m7: pretrain 6000, epochs 1000, enableat2000, lambda2 20, alpha 0.12, lr0.0002
- coil_m8: pretrain 5000, epochs 1000, enableat2000, lambda2 20, alpha 0.12, lr0.0002

- coil_m3raw: pretrain 3500, epochs 500, enableat1000, lambda2 20, alpha 0.12, COIL20RRRaw
- coil_m3dim: pretrain 3500, epochs 500, enableat1000, lambda2 20, alpha 0.12, COIL20RRdimension

- coil_k1: pretrain 3500, epochs 500, enable-at 1000, lambda2 20, alpha 0.12, kernel_size 5

# ORL exps

submean, one2one, no-uni-norm
all run with lr=0.001
orl_000:
orl_001:
orl_010:
orl_100:

all run with lr=0.0002
orl_000_2:
orl_001_2:
orl_010_2:
orl_100_2:

all run with lr=0.001, but enable-at 200
orl_000_3:
orl_001_3:
orl_010_3:
orl_100_3:

all run with lr=0.001, but enable-at 100
orl_000_4:
orl_001_4:
orl_010_4:
orl_100_4:

all run with lr=0.001, but enable-at 51
orl_000_5:
orl_001_5:
orl_010_5:
orl_100_5:

Without uniform normalization, training seem somewhat unstable.
There seems to be a bug in submean

So the best so far seems to be orl_010_4

orl_010_4_a0.05 to orl_010_4_a0.11 search for alpha. Turns out the default value 0f 0.1 was best

