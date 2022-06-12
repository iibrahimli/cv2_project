"""
Eye fixation dataset.
"""

import re
from pathlib import Path

import torch
import imageio
from torch.utils.data import Dataset


def _get_list_from_file(file_path):
    """
    Returns a list of lines from a file.
    """
    with open(file_path, "r") as f:
        lines = list(map(str.strip, f.readlines()))
    return lines


class FixationDataset(Dataset):
    def __init__(self, data_root_dir, split, transform=None):
        self.data_root_dir = Path(data_root_dir)

        all_splits = ("train", "val", "test")
        assert split in all_splits, f"Split must be one of {all_splits}"
        self.split = split

        self.transform = transform

        self.images = _get_list_from_file(self.data_root_dir / f"{split}_images.txt")

        # human sort the filenames
        int_re = re.compile(r"\d+")
        sort_key = lambda x: int(int_re.search(x).group())
        self.images.sort(key=sort_key)

        # fixations not available for test split
        if split != "test":
            self.fixations = _get_list_from_file(
                self.data_root_dir / f"{split}_fixations.txt"
            )
            self.fixations.sort(key=sort_key)

            assert len(self.images) == len(
                self.fixations
            ), "Number of images and fixations must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = imageio.imread(self.data_root_dir / self.images[idx])
        if self.split != "test":
            fix = imageio.imread(self.data_root_dir / self.fixations[idx])
            fix = fix / fix.max()
        else:
            fix = None

        if self.transform:
            aug = self.transform(image=img, mask=fix)
            img = aug["image"]
            fix = aug["mask"]

        fix = fix[None, ...]
        sample = {"image": img, "fixation": fix}

        return sample

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.
        """

        # batch is a list of dicts
        imgs = torch.stack([x["image"] for x in batch])
        fixs = torch.stack([x["fixation"] for x in batch])

        return {"image": imgs, "fixation": fixs}
