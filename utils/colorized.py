"""
Colorize depth map

Author: Foryoung Yu
Email: fuyang_yu@outlook.com
Date: 2023-07-13
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import cv2
from PIL import Image

parser = argparse.ArgumentParser(
    description="Colorize depth map", fromfile_prefix_chars="@"
)
# parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument("--input_dir", type=str, help="depth dir", default="input")
parser.add_argument("--output_dir", type=str, help="color dir", default="output")
parser.add_argument("--cmap", type=str, help="color map", default="magma_r")
parser.add_argument("--vmin", type=float, help="min value", default=None)
parser.add_argument("--vmax", type=float, help="max value", default=None)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = "@" + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


def colorize(value, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap):
    # value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.0

    cmapper = matplotlib.colormaps.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3] # remove alpha channel
    img = img[..., ::-1] # BGR to RGB
    
    return img


img_list = os.listdir(args.input_dir)
for img_name in img_list:
    img_path = os.path.join(args.input_dir, img_name)
    
    # depth_map = np.asarray(Image.open(img_path), dtype=np.float32)
    depth_map = cv2.imread(img_path, flags=2) # cv2.IMREAD_ANYDEPTH
    colorized = colorize(depth_map)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    # Image.fromarray(colorized).save(os.path.join(args.output_dir, img_name))
    cv2.imwrite(os.path.join(args.output_dir, img_name), colorized)
    
print("All Done.")
