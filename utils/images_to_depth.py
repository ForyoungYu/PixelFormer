"""

"""
import time
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

dataset = "nyu"
max_depth = 10
min_depth = 0
input_path = "input"
output_dir = "output"
img_list = os.listdir(input_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")  # Force use CPU
print("Device: {}".format(device))

# Define model
from pixelformer.networks.PixelFormer import PixelFormer

model = PixelFormer()

# Load pretrained model
ckpt_path = "checkpoints\model-117000-best_rms_0.32058.ckpt"
print("== Loading checkpoint '{}'".format(ckpt_path))
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model"])
del checkpoint

model.to(device)
model.eval()

# Transform images
def transform(img):
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((416, 544), interpolation=2)]
    )
    img = transf(img)
    return img.unsqueeze(0)  # (bxcxhxw)


for img_name in img_list:
    img_path = os.path.join(input_path, img_name)
    img = cv2.imread(img, cv2.IMREAD_ANYDEPTH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = Image.open(img_path)
    # depth_map = np.asarray(image, dtype=np.float32) / 255.0
    input = transform(img).to(device)
    start = time.time()

    # Prediction and resize to original resolution
    with torch.no_grad():
        pred = model(input)

        # Delete last dim if it exist
        pred = pred.squeeze(0)

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = pred.cpu().numpy()

    pred[np.isinf(pred)] = max_depth
    pred[np.isnan(pred)] = min_depth

    if dataset == "nyu":
        factor = 1000
    else:
        factor = 256
    pred = (pred * factor).astype("uint16")
    end = time.time()
    totalTime = end - start
    print("Process " + img_name + " total Time: %.2f s" % totalTime)

    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)  # Color mode
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    Image.fromarray(depth_map).save(os.path.join(output_dir, img_name))

print("All Done.")
