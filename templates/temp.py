import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torchinfo import torchinfo
canvas = torch.randn(size=(3,100,100))


class TempCnv(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.convs = [
            nn.Conv2d(3,256,3,1, padding=1),
            nn.Conv2d(256,64,3,1, padding=1),
            nn.Conv2d(64,64,3,1, padding=1),
            nn.Conv2d(64,256,3,1, padding=1),
        ]
    
    def forward(self,x: torch.Tensor):
        out = x
        for cnv in self.convs:
            out = cnv(out)
            print(out.shape)
        return
    
    
import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 1x1 branch (direct compression / feature mixing)
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        # 1x1 -> 3x3 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=1)
        )

        # 1x1 -> 5x5 branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2)
        )

        # pooling -> 1x1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        print(b1.shape)
        print(b2.shape)
        print(b3.shape)
        print(b4.shape)
        # channel-wise concat
        return torch.cat([b1, b2, b3, b4], dim=1)

if __name__ == "__main__":
    # model = TempCnv()
    # model.eval()
    
    # inp = torch.randn(3,100,100)
    # out = model(inp)
    
    # print()
    # print(torchinfo(model))

    # model = InceptionBlock(3)
    # model.eval()
    
    # inp = torch.randn(3,100,100)
    # out = model(inp)
    resnet = resnet50()
    