# -*- coding: utf-8 -*-
# This file originally came from detect_and_remove.py by pyImageSearch
# modified by sangkny to handle the dataset
# USAGE
# python deduplicate_data.py --dataset dataset
# python deduplicate_data.py --dataset dataset --remove 1

# import the necessary packages

import numpy as np
import argparse
import cv2
import os

# handle the os paths for file list
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = (resized[:, 1:] >= resized[:, :-1])

    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-r", "--remove", type=int, default=-1,
                help="whether or not duplicates should be removed (i.e., dry run)")
ap.add_argument("-s", "--show", type=bool, default=True,
                help="debug show image")
args = vars(ap.parse_args())

#args['dataset'] = "E:\\nexquad-ralated\\5cameras\\gather_images\\jpgs\\3channels\\vpd_images_model_v20200910_20200912_183335\\refined\\0"
args['dataset'] = "D:\\sangkny\\pyTest\\MLDL\\NexQuadDataSets\\3channels\\40x32\\refined\\1"
#args['dataset'] = "D:\\sangkny\\pyTest\\MLDL\\codes\\parkingClassify-master\\augimg_20200920_3channels_br04\\0"
args['remove'] = 1
debugImg = args['show']

# grab the paths to all images in our input dataset directory and
# then initialize our hashes dictionary
print("[INFO] computing image hashes...")
imagePaths = list(list_images(args["dataset"]))
hashes = {}

# loop over our image paths
for imagePath in imagePaths:
    # load the input image and compute the hash
    image = cv2.imread(imagePath)
    if(isinstance(image, type(None))):
        print(imagePath)
        continue
    h = dhash(image)

    # grab all image paths with that hash, add the current image
    # path to it, and store the list back in the hashes dictionary
    p = hashes.get(h, [])
    p.append(imagePath)
    hashes[h] = p

total_rm_files = 0;
# loop over the image hashes
for (h, hashedPaths) in hashes.items():
    # check to see if there is more than one image with the same hash
    if len(hashedPaths) > 1:
        # check to see if this is a dry run
        if args["remove"] <= 0:
            # initialize a montage to store all images with the same
            # hash
            montage = None

            # loop over all image paths with the same hash
            for p in hashedPaths:
                # load the input image and resize it to a fixed width
                # and height
                image = cv2.imread(p)
                image = cv2.resize(image, (150, 150))

                # if our montage is None, initialize it
                if montage is None:
                    montage = image

                # otherwise, horizontally stack the images
                else:
                    montage = np.hstack([montage, image])

            # show the montage for the hash
            if debugImg == True:
                print("[INFO] hash: {}".format(h))
                cv2.imshow("Montage", montage)
                cv2.waitKey(1)

        # otherwise, we'll be removing the duplicate images
        else:
            # loop over all image paths with the same hash *except*
            # for the first image in the list (since we want to keep
            # one, and only one, of the duplicate images)
            # print("[INFO] hash: {}".format(h))

            for p in hashedPaths[1:]:
                os.remove(p)
                total_rm_files += 1


print('[RES] %s were duplicated and removed. '%(total_rm_files))