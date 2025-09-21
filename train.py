import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from data_loader import EuroSATDataset
from model_resnet import build_resnet18
from utils import plot_training_curves
from tqdm import tqdm

def get_dataloaders(root, subset="EuroSAT", img_size=224, bs=64):
    data_dir = os.path.join(root, subset)
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    dataset = EuroSATDataset(data_dir, transform=tf_train)
    n = len(dataset)
    val_size = int(0.2 * n)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    # ensure val uses validation transform
    val_ds.dataset.transform = tf_val
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl, dataset

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        running_correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss/total, running_correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return running_loss/total, running_correct/total, np.concatenate(all_preds), np.concatenate(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/kaggle/input/eurosat-dataset", help="dataset root")
    parser.add_argument("--subset", default="EuroSAT", help="EuroSAT or EuroSATallBands")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, dataset = get_dataloaders(args.root, subset=args.subset, img_size=args.img_size, bs=args.bs)
    in_channels = 3
    num_classes = len(dataset.classes)

    model = build_resnet18(in_channels=in_channels, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_dl, criterion, device)
        elapsed = time.time() - start

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(f"Epoch {epoch}/{args.epochs} - train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f} - time: {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out, "best_model.pth"))
            print("✅ Best model saved.")

    # save curves
    plot_training_curves(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, args.out)
    print("Training complete. Best val acc:", best_acc)

if __name__ == "__main__":
    main()
