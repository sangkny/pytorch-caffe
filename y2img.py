# -*- coding: utf-8 -*-
# 20200213 by sangkny

# y2img : convert y image to normal format
# y2img(in_y_file, out_imgfile, img_height=40, img_width=32, debug_general = True)
# binary file i/o example

import os
import numpy as np
import cv2

def y2img(in_y_file, out_imgfile, img_height=40, img_width=32, debug_general = True, img_quality=100):

    file_height, file_width, file_debug = img_height, img_width, debug_general
    num_channels = 0
    if os.path.isfile(in_y_file):
        f = open(in_y_file, 'rb')
        #lines = f.readlines()
        lines = f.read()
        #indices = range(len(lines[0])) # when open with rt and f.readlines() -> list output
        indices = range(len(lines))

        if int(file_height*file_width) == len(lines): # gray
            num_channels = 1
            if(debug_general):
                print('this 1 channel : h x w:{}x{}'.format(file_height, file_width))

        elif int(file_height*file_width*3) == len(lines): # B/G/R 3 channels
            num_channels = 3
            if (debug_general):
                print('B/G/R 3 channel : h x w x 3:{}x{}x 3'.format(file_height, file_width))
        else:
            print('\n ---------------> error <---------------\n')
            print('file size is not correct: %s \n'% in_y_file)
            return 0
        if num_channels>1:
            img_nparray = np.zeros([file_height, file_width, num_channels], dtype='uint8')
        else:
            img_nparray = np.zeros([file_height, file_width], dtype='uint8')
        for i in indices:
            idx_row = int(i/file_width)       # it has row and channel information
            idx_ch = int(idx_row/file_height) # channel number  [0 num_channels-1]
            idx_row %= file_height            # converts only for image rows [0 file_height-1]
            idx_col = i%file_width            # converst only for image cols [0 file_width-1]
            if(debug_general and (idx_ch==0 and idx_col == 0 and idx_row == 0)):
                print("(row x col x ch: {}x{}x{})->({}) \n".format(idx_row,idx_col,idx_ch, lines[i]))
            if num_channels> 1:
                img_nparray[idx_row][idx_col][idx_ch] = lines[i]
            else: # 1 channel
                img_nparray[idx_row][idx_col]= lines[i]
        f.close()
        # need to convert B/G/R to R/G/B with channel swaps for processing.
        # however, if you want to write the image as RGB format using opencv , the order of channels should be
        # B/G/R. Then opencv write a file with the order of RGB
        # cv2.cvtColor(img_nparray,cv2.COLOR_BGR2RGB, img_nparray)
        if(debug_general):
            cv2.imshow('test', img_nparray)
            cv2.waitKey(1)

        if not cv2.imwrite(out_imgfile, img_nparray, [int(cv2.IMWRITE_JPEG_QUALITY), int(img_quality)]):
            print('Writing Error: %s' %out_imgfile)

    else:
        print('File No Exists: %s' %in_y_file)


