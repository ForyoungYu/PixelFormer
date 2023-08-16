import time
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

from models import FastDepth


input_path = 'input'
output_path = 'output'
img_list = os.listdir(input_path)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cpu')  # Force use CPU
print('Device: {}'.format(device))

# Define model
model = FastDepth()

# Load pretrained model
ckpt = 'saved_models\FastDepth_nyu_06-Nov_20-12-bs20-tep300-lr0.000357-wd0.1_best.pt'
model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
model.to(device)
model.eval()

def colorize(value, vmin=None, vmax=None, cmap='magma_r'):
    value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    print(vmin, vmax)
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(0)
    cmapper = mpl.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # nxmx4

    img = value[:, :, :3]

    # return img.transpose((2, 0, 1))
    return img

# Transform images
def transform(img):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 544), interpolation=2)
    ])
    img = transf(img) 
    return img.unsqueeze(0)  # (bxcxhxw)

for img_name in img_list:
    img_path = os.path.join(input_path, img_name)
    
    img = np.asarray(Image.open(img_path), dtype=np.float32) / 255.0

    input = transform(img).to(device)
    start = time.time()

    # Prediction and resize to original resolution
    with torch.no_grad():
        pred = model(input)

        # Delete last dim if it exist
        pred = pred.squeeze(0)

    # print(pred.shape)
    # exit()
    end = time.time()
    totalTime = end - start
    print("Process " + img_name + " total Time: %.2f s" % totalTime)
    depth_map = colorize(pred)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_name = os.path.join(output_path, img_name)
    cv2.imwrite(output_name, depth_map)
    # exit()

print("All Done.")