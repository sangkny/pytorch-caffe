
# y2img batch program..
# In a folder, all y images will be converted into formatted images.

import os
import cv2
import sys
import argparse

from y2img import y2img as yToimg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch Conversion from Y to Image')
    parser.add_argument('--srcDir', default='C:\\Users\\mmc\\Downloads\\20200212_test_result\\1', type=str, help='base directory')
    parser.add_argument('--tgtDir', default='C:\\Users\\mmc\\Downloads\\20200212_test_result\\1', type=str, help='target directory')
    parser.add_argument('--img_height', default=40, type=int, help='y image size: height')
    parser.add_argument('--img_width', default=32, type=int, help='y image size: width')
    parser.add_argument('--srcExt', default='y', type=str, help='input extension')
    parser.add_argument('--tgtExt', default='jpg', type=str, help='target extension')
    parser.add_argument('--showImg', default=True, type=bool, help='flag to show images during process')
    parser.add_argument('--saveImg', default=True, type=bool, help=' flag to save generated images')
    args = parser.parse_args()

    # parameter settings
    srcDir = args.srcDir
    tgtDir = args.tgtDir
    img_height = args.img_height
    img_width = args.img_width
    debug_showImg = args.showImg
    debug_saveImg = args.saveImg

    in_file_ext = args.srcExt #'y'
    out_file_ext = args.tgtExt #'jpg'
    src_full_path = srcDir+'/*.'+in_file_ext    # find all data

    files = os.listdir(srcDir)

    for file in files:
        if(str(file.split('.')[-1]).lower() == str(in_file_ext).lower()):
            new_file = file.split('.')[:-1][0] + '.' + out_file_ext
            src_full_path = srcDir + '/' + file
            tgt_full_path = tgtDir + '/' + new_file
            if not os.path.isdir(tgtDir):
                os.mkdir(tgtDir)

            # y2img(in_y_file, out_imgfile, img_height=40, img_width=32, debug_general = True)
            yToimg(in_y_file=src_full_path, out_imgfile=tgt_full_path, img_height=img_height, img_width=img_width, debug_general=debug_showImg)

