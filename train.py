"""
Training script.

TODO:
"""

import time
import argparse
from pathlib import Path

import wandb
import torch
import torch.nn as nn
import albumentations as A
from torchvision.utils import make_grid
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from fixation import FixationDataset, FixNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_dataloader(
    data_root_dir,
    split,
    batch_size,
    use_aug=False,
    shuffle=True,
):
    transforms = []
    if use_aug:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.3),
                A.Rotate(45, p=0.3),
                A.ShiftScaleRotate(p=0.3),
                A.OpticalDistortion(p=0.1),
                A.RandomBrightnessContrast(0.2, 0.15, p=0.5),
                A.HueSaturationValue(p=0.5),
            ]
        )
    transforms.extend(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    transform = A.Compose(transforms)
    return DataLoader(
        dataset=FixationDataset(
            data_root_dir=data_root_dir,
            split=split,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=FixationDataset.collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )


def get_prediction_demo(batch, model):
    """Returns image grid of predictions."""
    imgs = batch["image"][:16].to(device)
    labels = batch["fixation"][:16]

    model.eval()
    with torch.no_grad():
        preds = model(imgs)
    preds = model(imgs).cpu().detach().repeat(1, 3, 1, 1)
    labels = labels.repeat(1, 3, 1, 1)

    # de-normalize images
    imgs = imgs.cpu().detach()
    means = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    stds = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    imgs = (imgs * stds) + means
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

    # scale preds to [0, 1]
    preds = torch.exp(preds)
    preds = preds / preds.max()

    # scale labels
    labels = labels / labels.max()

    # merge images, labels, and predictions
    res = torch.cat((imgs, labels, preds), dim=3)
    res = make_grid(res, nrow=2)

    return res


def run_on_batch(model, batch, loss_fn):
    img = batch["image"].to(device)
    fix = batch["fixation"].to(device)
    pred = model(img)
    loss = loss_fn(pred, fix)
    return pred, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_warmup", type=int, default=-1)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_center_bias", action="store_true")
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--es_patience", type=int, default=100)
    args = parser.parse_args()

    print("Using args:")
    for arg in vars(args):
        print(f"    {arg}: {getattr(args, arg)}")

    cp_dir = Path(args.checkpoint_dir)
    cp_dir.mkdir(exist_ok=True)
    cp_path = cp_dir / f"model.pth"

    # get dataloaders
    train_dl = get_dataloader(
        args.data_root_dir, "train", batch_size=args.batch_size, use_aug=not args.no_aug
    )
    val_dl = get_dataloader(args.data_root_dir, "val", batch_size=args.batch_size)
    test_dl = get_dataloader(args.data_root_dir, "test", batch_size=args.batch_size)

    print(f"Training: {len(train_dl) * args.batch_size} images")
    print(f"Validation: {len(val_dl) * args.batch_size} images")
    print(f"Testing: {len(test_dl) * args.batch_size} images")

    # initialize model, loss, and optimizer
    center_bias_path = None
    if not args.no_center_bias:
        center_bias_path = Path(args.data_root_dir) / "center_bias_density.npy"
    model = FixNet(
        center_bias_path=center_bias_path,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_trainable_params}")

    if args.resume:
        cp = torch.load(cp_path)
        model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        print("Loaded state from checkpoint")

    if not args.no_wandb:
        wandb.init(project="fixation-prediction", config=args)
        wandb.watch(model, log_freq=args.log_freq * 10)

    # demo batch
    demo_batch = next(iter(val_dl))

    step = -1
    min_val_loss = float("inf")
    es_counter = 0
    stopped = False

    # training loop
    for epoch in range(args.max_epochs):
        if stopped:
            break

        for it, batch in enumerate(train_dl):
            step += 1

            if (
                args.freeze_encoder
                and args.freeze_warmup > 0
                and step == args.freeze_warmup
            ):
                model.set_encoder_trainable(True)
                print("Thawed the encoder")

            start_time = time.perf_counter()

            model.train()
            optimizer.zero_grad()
            pred, loss = run_on_batch(model, batch, loss_fn)
            loss.backward()
            optimizer.step()

            it_time = round((time.perf_counter() - start_time) * 1000)

            log_prefix = f"[epoch {epoch+1}/{args.max_epochs}, iter {it+1}/{len(train_dl)}, step {step}]"

            # log
            if step % args.log_freq == 0:
                print(f"{log_prefix} train loss: {loss.item():.4f} ({it_time:} ms/it)")
                if not args.no_wandb:
                    wandb.log({"loss": loss.item()}, step=step)

            # validation
            if step % args.val_freq == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for it, batch in enumerate(val_dl):
                        _, loss = run_on_batch(model, batch, loss_fn)
                        val_loss += loss.item()
                val_loss = val_loss / len(val_dl)

                # log
                print(f"{log_prefix} val loss: {val_loss:.4f}")
                if not args.no_wandb:
                    demo_imgs = get_prediction_demo(demo_batch, model)
                    wandb.log(
                        {
                            "val_loss": val_loss,
                            "demo_imgs": wandb.Image(
                                demo_imgs, caption="image, label, prediction"
                            ),
                        },
                        step=step,
                    )

                # early stopping
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    es_counter = 0
                    print(f"{log_prefix} New minimum validation loss: {val_loss:.4f}")
                    # save checkpoint
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        cp_path,
                    )
                    print(f"Saved checkpoint to {cp_path}")
                else:
                    es_counter += 1
                    if es_counter >= args.es_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        stopped = True
                        break
