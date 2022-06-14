"""
Evaluation on the test set
"""

import argparse
from pathlib import Path
import imageio

import torch
from tqdm import tqdm
import albumentations as A
from torchvision.utils import make_grid
from torchvision.transforms import ConvertImageDtype
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
                A.Rotate(45, p=0.25),
                A.ShiftScaleRotate(p=0.3),
                A.OpticalDistortion(p=0.1),
                A.RandomBrightnessContrast(0.2, 0.15, p=0.3),
                A.HueSaturationValue(p=0.2),
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


def get_prediction_demo(batch, model, nrow):
    """Returns image grid of predictions."""
    imgs = batch["image"].to(device)
    labels = batch["fixation"]

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
    res = make_grid(res, nrow=nrow)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_root_dir", type=str, default="data")
    args = parser.parse_args()

    print("Using args:")
    for arg in vars(args):
        print(f"    {arg}: {getattr(args, arg)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    test_dl = get_dataloader(args.data_root_dir, "test", batch_size=1, use_aug=False)

    model = FixNet(
        center_bias_path=None,
        freeze_encoder=False,
    ).to(device)

    cp = torch.load(args.model_path, map_location=device)
    model.load_state_dict(cp["model_state_dict"])
    print("Loaded model")

    losses = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_dl):
            img = batch["image"].to(device)
            output_path = batch["output_name"][0]
            pred = model(img).squeeze()
            pred = torch.sigmoid(pred)
            # max normalize to 1
            pred /= pred.max()
            out = ConvertImageDtype(torch.uint8)(pred).numpy()
            imageio.imsave(output_dir / output_path, out)