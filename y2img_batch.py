
# y2img batch program..
# In a folder, all y images will be converted into formatted images.

import os
import cv2
import sys

from y2img import y2img as yToimg

in_file = './data/1.y'
out_file = './data/1_out.jpg'

yToimg(in_file, out_file, 40, 32, True)

