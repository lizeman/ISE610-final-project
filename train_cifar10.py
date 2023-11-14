import argparse
import wandb
import torch
import json
import numpy as np
import random
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch import optim
import torch.nn as nn
from model import WRN_16_4
from data import select_train_valid_loader, select_test_loader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_per_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_acc += predicted.eq(targets).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_acc


def test_per_epoch(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_acc += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    return test_loss, test_acc


def experiment(args):
    setup_seed(args.seed)
    device = torch.device(args.device)

    # load model
    model = WRN_16_4().to(device)
    model = ModuleValidator.fix(model)

    # load data
    train_loader, val_loader = select_train_valid_loader(args)
    test_loader = select_test_loader(args)

    # load optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # load privacy engine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        poisson_sampling=False,
    )

    # train
    for epoch in range(args.epochs):
        train_loss, train_acc = train_per_epoch(
            model, optimizer, criterion, train_loader, device
        )
        val_loss, val_acc = test_per_epoch(model, criterion, val_loader, device)
        test_loss, test_acc = test_per_epoch(model, criterion, test_loader, device)
        print(
            f"Epoch {epoch}: train loss: {train_loss : 5f}, train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc}, test loss: {test_loss}, test acc: {test_acc}"
        )

    # Evaluate
    test_loss, test_acc = test_per_epoch(model, criterion, test_loader, device)
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cpu", help="choose device type: cpu, cuda, mps"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_n_worker", type=int, default=0, help="number of worker for dataloader"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./raw_data",
        help="location where data is stored",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="batch size used in training"
    )
    parser.add_argument(
        "--train_valid_ratio",
        type=float,
        default=0.2,
        help="ratio of train and validation dataset",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=25.8,
        help="target epsilon for differential privacy",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="max norm for gradient clipping",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-6,
        help="target delta for differential privacy",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="the epochs for performing private training",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")


    args = parser.parse_args()
    print(f"Running on {args.device}")
    experiment(args)


if __name__ == "__main__":
    main()
