from typing import Any
import pandas as pd
from train_cifar10 import experiment
import os
import argparse


class dummy_args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main():
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    curr_args = parser.parse_args()
    output_filename = f"result_{curr_args.start}_{curr_args.end}.csv"

    df = pd.read_csv("optimal.csv")[curr_args.start : curr_args.end]
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(output_dir, output_filename))
    for i in range(len(df)):
        args = dummy_args()
        args.device = "cuda"
        args.data_n_worker = 0
        args.data_dir = "./raw_data"
        args.batch_size = 2 ** int(df.loc[i, "lg2(batch_size)"])
        args.train_valid_ratio = 0.2
        args.epsilon = 25.8
        args.max_grad_norm = df.loc[i, "clipping_threshold"]
        args.delta = 1e-5
        args.epochs = int(df.loc[i, "epochs"])
        args.seed = 42
        args.lr = df.loc[i, "learning_rate"]
        args.momentum = df.loc[i, "momentum"]
        print(args.__dict__)
        test_loss, test_acc = experiment(args)
        df.loc[i, "Accuracy"] = test_acc
        df.loc[i, "Loss"] = test_loss
        df.to_csv(os.path.join(output_dir, output_filename))
    df.to_csv(os.path.join(output_dir, output_filename))

if __name__ == "__main__":
    main()
