import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fixation import config, FixationDataset, FixNet


def get_dataloader(
    split,
    batch_size,
    image_transform=None,
    fixation_transform=None,
    shuffle=True,
    num_workers=0,
):
    return DataLoader(
        dataset=FixationDataset(
            root_dir=config.DATA_ROOT_DIR,
            split=split,
            image_transform=image_transform,
            fixation_transform=fixation_transform,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


train_dl = get_dataloader("train", batch_size=config.BATCH_SIZE, shuffle=True)
val_dl = get_dataloader("val", batch_size=config.BATCH_SIZE, shuffle=True)
test_dl = get_dataloader("test", batch_size=config.BATCH_SIZE, shuffle=True)

model = FixNet()
model(torch.randn(1, 3, 224, 224))