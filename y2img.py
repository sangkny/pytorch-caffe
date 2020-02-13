#
# y2img : convert y image to normal format
import os
import numpy as np
import cv2

def y2img(in_y_file, out_imgfile, img_height=40, img_width=32, debug_general = True):

    file_height, file_width, file_debug = img_height, img_width, debug_general

    if os.path.isfile(in_y_file):
        f = open(in_y_file, 'rb')
        lines = f.readlines()
        img_nparray = np.zeros([file_height, file_width], dtype ='uint8')
        indices = range(len(lines[0]))

        if int(file_height*file_width) != len(lines[0]):
            print('\n ---------------> error <---------------\n')
            print('file size is not correct: %s \n'% in_y_file)
            return 0

        for i in indices:
            idx_row = int(i/file_width)
            idx_col = i%file_width
            if(debug_general):
                print("(row: {}, col: {})->({}) \n".format(idx_row,idx_col, lines[0][i]))
            img_nparray[idx_row][idx_col] = lines[0][i]
        f.close()

        if(debug_general):
            cv2.imshow('test', img_nparray)
            cv2.waitKey(0)

        cv2.imwrite(out_imgfile, img_nparray)

    else:
        print('File No Exists: %s' %in_y_file)


