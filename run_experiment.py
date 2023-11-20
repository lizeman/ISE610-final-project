from typing import Any
import pandas as pd
from train_cifar10 import experiment


class dummy_args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main():
    df = pd.read_csv("result_remain.csv")
    df = df.reset_index(drop=True)
    df.to_csv("result_Nov20.csv")
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
        args.weight_decay = df.loc[i, "weight_decay"]
        print(args.__dict__)
        if args.batch_size == 2 ** 12:
            continue
        test_loss, test_acc = experiment(args)
        df.loc[i, "Accuracy"] = test_acc
        df.loc[i, "Loss"] = test_loss
        df.to_csv("result_Nov20.csv")
    df.to_csv("result_Nov20.csv")


if __name__ == "__main__":
    main()
