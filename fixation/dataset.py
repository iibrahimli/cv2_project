"""
Eye fixation dataset.
"""

import re
from pathlib import Path

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
    def __init__(self, root_dir, split, image_transform=None, fixation_transform=None):
        self.root_dir = Path(root_dir)

        all_splits = ("train", "val", "test")
        assert split in all_splits, f"Split must be one of {all_splits}"
        self.split = split

        self.image_transform = image_transform
        self.fixation_transform = fixation_transform

        self.images = _get_list_from_file(self.root_dir / f"{split}_images.txt")

        # human sort the filenames
        int_re = re.compile(r"\d+")
        sort_key = lambda x: int(int_re.search(x).group())
        self.images.sort(key=sort_key)

        # fixations not available for test split
        if split != "test":
            self.fixations = _get_list_from_file(
                self.root_dir / f"{split}_fixations.txt"
            )
            self.fixations.sort(key=sort_key)

            assert len(self.images) == len(
                self.fixations
            ), "Number of images and fixations must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = imageio.imread(self.root_dir / self.images[idx])
        if self.split != "test":
            fix = imageio.imread(self.root_dir / self.fixations[idx])
        else:
            fix = None

        sample = {"image": img, "fixation": fix}

        if self.image_transform:
            sample["image"] = self.image_transform(sample["image"])
        if self.split != "test" and self.fixation_transform:
            sample["fixation"] = self.fixation_transform(sample["fixation"])

        return sample
