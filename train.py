import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fixation import config, FixationDataset, FixNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
loss_fcn = nn.MSELoss()

wandb.init(project="fixation-prediction", config=config)
wandb.watch(model)

# training loop
for epoch in range(config.EPOCHS):
    pass