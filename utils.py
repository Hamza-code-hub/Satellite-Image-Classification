import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import imageio

def plot_confusion_matrix(cm, classes, outpath):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_training_curves(train_loss, val_loss, train_acc, val_acc, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(train_acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

def save_sample_grid(img_tensors, labels, preds, class_names, outpath, ncols=4):
    """
    img_tensors: list of torch tensors (C,H,W) normalized in [-]
    labels/preds: lists
    """
    import torch
    n = len(img_tensors)
    nrows = (n + ncols - 1)//ncols
    plt.figure(figsize=(4*ncols, 4*nrows))
    for i in range(n):
        plt.subplot(nrows, ncols, i+1)
        img = img_tensors[i].cpu().permute(1,2,0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        color = "green" if labels[i]==preds[i] else "red"
        plt.title(f"T:{class_names[labels[i]]}\nP:{class_names[preds[i]]}", color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def build_gif_from_frames(frame_dir, output_path, fps=2):
    files = sorted([os.path.join(frame_dir,f) for f in os.listdir(frame_dir) if f.endswith(".png")])
    if not files:
        print("No frames found:", frame_dir)
        return
    imgs = [imageio.imread(f) for f in files]
    imageio.mimsave(output_path, imgs, fps=fps)
    print("GIF saved:", output_path)
