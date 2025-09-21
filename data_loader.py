import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class EuroSATDataset(Dataset):
    """EuroSAT RGB dataset loader (folder->class)."""
    def __init__(self, root, transform=None, csv_file=None):
        """
        root: path to dataset root, e.g. '/kaggle/input/eurosat-dataset/EuroSAT'
        transform: torchvision transforms
        csv_file: optional CSV with columns path,label relative to root/subset
        """
        self.root = root
        self.transform = transform
        self.samples = []

        # If CSV provided, use it
        if csv_file and os.path.exists(csv_file):
            import pandas as pd
            df = pd.read_csv(csv_file)
            # Support (path,label) columns or (filename,label)
            if 'path' in df.columns and 'label' in df.columns:
                for _, r in df.iterrows():
                    p = os.path.join(root, r['path'])
                    self.samples.append((p, int(r['label'])))
            else:
                # assume two columns: filename,label
                for _, r in df.iterrows():
                    p = os.path.join(root, str(r.iloc[0]))
                    self.samples.append((p, int(r.iloc[1])))
        else:
            # scan subdirectories (ignore files like train.csv)
            classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for c in classes:
                cdir = os.path.join(root, c)
                for fname in os.listdir(cdir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(cdir, fname), self.class_to_idx[c]))
        # make class list attribute for convenience
        if hasattr(self, 'class_to_idx'):
            self.classes = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]
        else:
            # derive from labels
            labels = sorted(list({lbl for _, lbl in self.samples}))
            self.classes = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(label)
