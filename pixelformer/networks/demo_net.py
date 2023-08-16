import torch.nn as nn


class DemoNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(DemoNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample((530, 730), mode="bilinear", align_corners=False),
            nn.Conv2d(32, out_channels, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = self.up1(out)
        out = self.up2(out)

        return out
