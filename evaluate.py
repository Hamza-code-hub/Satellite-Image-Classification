import os
import argparse
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import EuroSATDataset
from model_resnet import build_resnet18
from utils import plot_confusion_matrix, save_sample_grid, build_gif_from_frames
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def get_loader(root, subset="EuroSAT", img_size=224, bs=64):
    data_dir = os.path.join(root, subset)
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = EuroSATDataset(data_dir, transform=tf)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    return ds, loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/kaggle/input/eurosat-dataset")
    parser.add_argument("--subset", default="EuroSAT")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, loader = get_loader(args.root, args.subset, args.img_size, args.bs)

    model_path = os.path.join(args.out, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("best_model.pth not found in outputs/ — run train.py first.")

    model = build_resnet18(in_channels=3, num_classes=len(ds.classes), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    start = time.time()
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    elapsed = time.time() - start
    all_probs = np.vstack(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=ds.classes))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, ds.classes, os.path.join(args.out, "confusion_matrix.png"))

    # ROC curves (one-vs-rest)
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(len(ds.classes)))
    n_classes = y_true_bin.shape[1]

    plt.figure(figsize=(10,8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{ds.classes[i]} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.out, "roc_curves.png"))
    plt.close()

    # inference time per image
    avg_time = elapsed / len(ds)
    print(f"Inference time (avg) per image: {avg_time*1000:.2f} ms")

    # save sample predictions grid and frames for GIF
    # pick 12 random samples
    import random
    idxs = random.sample(range(len(ds)), k=12)
    imgs = []
    labels = []
    preds = []
    frame_dir = os.path.join(args.out, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    from PIL import Image
    for i, idx in enumerate(idxs):
        img, label = ds[idx]
        imgs.append(img)
        labels.append(label)
        # run through model single
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))
            pred = out.argmax(1).item()
            preds.append(pred)
        # save frame for gif
        # unnormalize
        img_disp = img.permute(1,2,0).cpu().numpy()
        img_disp = (img_disp - img_disp.min()) / (img_disp.max()-img_disp.min())
        plt.imshow(img_disp)
        color = "green" if label==preds[-1] else "red"
        plt.title(f"T:{ds.classes[label]} | P:{ds.classes[preds[-1]]}", color=color)
        plt.axis("off")
        fname = os.path.join(frame_dir, f"frame_{i:03d}.png")
        plt.savefig(fname)
        plt.clf()

    # save grid
    save_sample_grid(imgs, labels, preds, ds.classes, os.path.join(args.out, "sample_predictions.png"))

    # combine frames into gif (user can run make_gif.py or we can build here)
    build_gif_from_frames(frame_dir, os.path.join(args.out, "predictions.gif"), fps=1)

    print("Evaluation complete. Outputs saved to", args.out)

if __name__ == "__main__":
    main()
