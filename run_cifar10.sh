#!/bin/bash
export PYTHONPATH=$PWD

python train_cifar10.py  --seed 42 \
--device cuda \
--max_grad_norm 1 \
--epsilon 23.56 \
--delta 1e-5 \
--epochs 126 \
--lr 0.85867