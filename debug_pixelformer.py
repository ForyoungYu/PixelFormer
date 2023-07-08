import torch
import torch.nn as nn

from pixelformer.networks.PixelFormer import PixelFormer

x = torch.randn(1, 3, 224, 224)

model = PixelFormer(
    # backbone="biformer_base",
    version="large07",
    pretrained="pretrained/swin_large_patch4_window7_224_22k.pth",
    min_depth=0.1,
    max_depth=100.0,
)
out = model(x)
print(out.shape)
