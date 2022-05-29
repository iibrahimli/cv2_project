import time
import argparse
from pathlib import Path

import wandb
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from fixation import FixationDataset, FixNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_dataloader(
    root_dir,
    split,
    batch_size,
    shuffle=True,
    num_workers=0,
):
    aug = A.Compose(
        [
            # A.HorizontalFlip(always_apply=True),
            A.RandomBrightnessContrast(0.3, 0.2, p=0.5),
            # A.Rotate(limit=45, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return DataLoader(
        dataset=FixationDataset(
            root_dir=root_dir,
            split=split,
            transform=aug,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=FixationDataset.collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )


def run_on_batch(model, batch, loss_fn):
    img = batch["image"].to(device)
    fix = batch["fixation"].to(device)
    pred = model(img)
    loss = loss_fn(pred, fix)
    return pred, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1)
    args = parser.parse_args()

    print("Using args:")
    for arg in vars(args):
        print(f"    {arg}: {getattr(args, arg)}")

    cp_dir = Path(args.checkpoint_dir)
    cp_dir.mkdir(exist_ok=True)

    # get dataloaders
    train_dl = get_dataloader(args.root_dir, "train", batch_size=args.batch_size)
    val_dl = get_dataloader(args.root_dir, "val", batch_size=args.batch_size)
    test_dl = get_dataloader(args.root_dir, "test", batch_size=args.batch_size)

    # initialize model, loss, and optimizer
    model = FixNet(
        Path(args.root_dir) / "center_bias_density.npy",
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_trainable_params}")

    if not args.no_wandb:
        wandb.init(project="fixation-prediction", config=args)
        wandb.watch(model)

    step = 0

    # training loop
    for epoch in range(args.max_epochs):
        for it, batch in enumerate(train_dl):

            start_time = time.perf_counter()

            model.train()
            optimizer.zero_grad()
            pred, loss = run_on_batch(model, batch, loss_fn)
            loss.backward()
            optimizer.step()

            it_time = round((time.perf_counter() - start_time) * 1000)

            # log
            if step % args.log_freq == 0:
                print(
                    f"[epoch {epoch+1}/{args.max_epochs}, iter {it+1}/{len(train_dl)}] train loss: {loss.item():.4f} ({it_time:} ms/it)"
                )
                if not args.no_wandb:
                    wandb.log({"loss": loss.item()}, step=step)

            step += 1

        # validation
        model.eval()
        with torch.no_grad():
            for it, batch in enumerate(val_dl):
                pred, loss = run_on_batch(model, batch, loss_fn)
                print(f"[epoch {epoch+1}/{args.max_epochs}] val loss: {loss.item():.4f}")
                if not args.no_wandb:
                    wandb.log({"val_loss": loss.item()}, step=step)

        # save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            cp_dir / f"latest.pt",
        )
