import torchvision.models as models
import torch.nn as nn
import torch

def build_resnet18(in_channels=3, num_classes=10, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    if in_channels != 3:
        # replace first conv
        old = model.conv1
        newconv = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            # init by averaging existing weights
            w = old.weight.data  # (out_c, 3, k, k)
            w_mean = w.mean(dim=1, keepdim=True)  # (out_c,1,k,k)
            new_w = w_mean.repeat(1, in_channels, 1, 1) * (3.0/in_channels)
            newconv.weight.data = new_w
        model.conv1 = newconv
    # replace final fc
    nfc = model.fc.in_features
    model.fc = nn.Linear(nfc, num_classes)
    return model
