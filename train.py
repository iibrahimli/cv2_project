import argparse

import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from fixation import FixationDataset, FixNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_dataloader(
    root_dir,
    split,
    batch_size,
    image_transform=None,
    fixation_transform=None,
    shuffle=True,
    num_workers=0,
):
    return DataLoader(
        dataset=FixationDataset(
            root_dir=root_dir,
            split=split,
            image_transform=image_transform,
            fixation_transform=fixation_transform,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=FixationDataset.collate_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    print("Using args:")
    for arg in vars(args):
        print(f"    {arg}: {getattr(args, arg)}")
    print()

    # get dataloaders
    train_dl = get_dataloader(args.root_dir, "train", batch_size=args.batch_size)
    val_dl = get_dataloader(args.root_dir, "val", batch_size=args.batch_size)
    test_dl = get_dataloader(args.root_dir, "test", batch_size=args.batch_size)

    # initialize model, loss, and optimizer
    model = FixNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fcn = nn.MSELoss()

    if not args.no_wandb:
        wandb.init(project="fixation-prediction", config=args)
        wandb.watch(model)

    # training loop
    for epoch in range(args.max_epochs):
        model(torch.randn(1, 3, 224, 224))
        n_params = sum(p.numel() for p in model.parameters())