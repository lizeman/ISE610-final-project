#!/bin/bash
export PYTHONPATH=$PWD

python train_cifar10.py  --seed 42 \
--device cuda \
--max_grad_norm 1.0 \
--epsilon 16 \
--delta 1e-5 \
--epochs 50 \
--lr 2