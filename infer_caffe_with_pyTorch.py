# -*- coding: utf-8 -*-
# 2020. 02. 12 sangkny for verifying the Nexquad Lenet performance
import sys
from caffenet import *
import numpy as np
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time


def load_image(imgfile, color_input):
    input_color = color_input
    image = caffe.io.load_image(imgfile, color=input_color)
    if input_color:
        transformer = caffe.io.Transformer(
            {'data': (1, 3, args.height, args.width)})  # sangkny was (1,3, args.height, args.width) is color image
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
        # transformer.set_mean('data', np.array([128]))
        transformer.set_raw_scale('data', args.scale)
        transformer.set_channel_swap('data', (2, 1, 0))
        image = transformer.preprocess('data', image)
        image = image.reshape(1, 3, args.height, args.width)
    else:
        transformer = caffe.io.Transformer(
            {'data': (1, 1, args.height, args.width)})  # sangkny was (1,3, args.height, args.width) is color image
        # transformer.set_transpose('data', (2, 0, 1))
        # transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
        transformer.set_mean('data', np.array([args.meanB]))
        transformer.set_raw_scale('data', args.scale)
        # transformer.set_channel_swap('data', (2, 1, 0))
        image = transformer.preprocess('data', image)
        image = image.reshape(1, 1, args.height, args.width)

    return image


def load_synset_words(synset_file):
    lines = open(synset_file).readlines()
    synset_dict = dict()
    for i, line in enumerate(lines):
        synset_dict[i] = line.strip()
    return synset_dict


def forward_pytorch(protofile, weightfile, image):
    net = CaffeNet(protofile, width=args.width, height=args.height, channels=1, omit_data_layer=True, phase='TEST')
    if args.cuda:
        net.cuda()
    #print(net)
    net.load_weights(weightfile)
    net.eval()
    image = torch.from_numpy(image)
    if args.cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)
    t0 = time.time()
    blobs = net(image)
    t1 = time.time()
    return t1 - t0, blobs, net.models, net

def Caffe2PytorchNet(protofile, weightfile):
    net = CaffeNet(protofile, width=args.width, height=args.height, channels=1, omit_data_layer=True, phase='TEST')
    if args.cuda:
        net.cuda()
    #print(net)
    net.load_weights(weightfile)
    net.eval()

    return net


# Reference from:
def forward_caffe(protofile, weightfile, image):
    if args.cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['data'].reshape(1, 1, args.height, args.width)
    net.blobs['data'].data[...] = image
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1 - t0, net.blobs, net.params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('--protofile', default='', type=str)
    parser.add_argument('--weightfile', default='', type=str)
    parser.add_argument('--imgfile', default='', type=str)
    parser.add_argument('--height', default=224, type=int)
    parser.add_argument('--width', default=224, type=int)
    parser.add_argument('--meanB', default=104, type=float)
    parser.add_argument('--meanG', default=117, type=float)
    parser.add_argument('--meanR', default=123, type=float)
    parser.add_argument('--scale', default=255, type=float)
    parser.add_argument('--synset_words', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()
    print(args)
    # ---- parameter settings ---------------
    output_layer = "fc_blob3" # correct the proper output layer, default:'prob'
    protofile = args.protofile
    weightfile = args.weightfile
    #imgfile =  'C:\\Users\\mmc\\Downloads\\20200214_test_result\\1\\vpdImage_19701010_132301_7.jpg' #args.imgfile
    imgfile = './data/cat.jpg'
    model_height = args.height
    model_width = args.width
    color_input = True
    describe_layer = False
    # ----------------------------------------------------

    image = load_image(imgfile, color_input)
    # convert the image according to the proper input for protofile
    # 1 channel
    # convert already bgr in previous step
    # gray = np.dot(image[...,:3],[0.114, 0.587, 0.299])
    image = image[0, 0, :, :]

    # --------------------show the given image
    import cv2
    cv2.imshow('test', np.array(image/255))
    cv2.waitKey(1000)
    # ----------------------------------
    image = image.reshape(1, 1, model_height, model_width)

    #time_pytorch, pytorch_blobs, pytorch_models, pytorch_net = forward_pytorch(protofile, weightfile, image)
    pytorch_net = Caffe2PytorchNet(protofile, weightfile)
    pytorch_models = pytorch_net.models

    image = torch.from_numpy(image)
    if args.cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)
    t0 = time.time()
    pytorch_blobs = pytorch_net(image)
    t1 = time.time()

    time_pytorch = t1-t0

    print('\n pytorch net \n')
    print(pytorch_net)
    print('pytorch forward time: %f msec' % (time_pytorch*1000))

    layer_names = pytorch_models.keys()
    blob_names = pytorch_blobs.keys()
    if describe_layer:
        print('------------ Parameter Description ------------')
        for layer_name in layer_names:
            if type(pytorch_models[layer_name]) in [nn.Conv2d, nn.Linear, Scale, Normalize]:
                pytorch_weight = pytorch_models[layer_name].weight.data
                if args.cuda:
                    pytorch_weight = pytorch_weight.cpu().numpy()
                else:
                    pytorch_weight = pytorch_weight.numpy()
                # weight_diff = abs(pytorch_weight - caffe_weight).sum()
                if type(pytorch_models[layer_name].bias) == Parameter:
                    pytorch_bias = pytorch_models[layer_name].bias.data
                    if args.cuda:
                        pytorch_bias = pytorch_bias.cpu().numpy()
                    else:
                        pytorch_bias = pytorch_bias.numpy()
                    # caffe_bias = caffe_params[layer_name][1].data
                    # bias_diff = abs(pytorch_bias - caffe_bias).sum()
                    print('%-30s ' % (layer_name))
                else:
                    print('%-30s ' % (layer_name))
            elif type(pytorch_models[layer_name]) == nn.BatchNorm2d:
                if args.cuda:
                    pytorch_running_mean = pytorch_models[layer_name].running_mean.cpu().numpy()
                    pytorch_running_var = pytorch_models[layer_name].running_var.cpu().numpy()
                else:
                    pytorch_running_mean = pytorch_models[layer_name].running_mean.numpy()
                    pytorch_running_var = pytorch_models[layer_name].running_var.numpy()

                print('%-30s running_mean: %f running_var: %f' % (layer_name, pytorch_running_mean.sum(), pytorch_running_var.sum()))
        print('------------ Output Difference ------------')
        for blob_name in blob_names:
            if args.cuda:
                pytorch_data = pytorch_blobs[blob_name].data.cpu().numpy()
            else:
                pytorch_data = pytorch_blobs[blob_name].data.numpy()

            print('%-30s pytorch_shape: %-20s ' % (blob_name, pytorch_data.shape))

    if args.synset_words != '':
        print('------------ Classification ------------')
        synset_dict = load_synset_words(args.synset_words)

        if output_layer in blob_names:
            if args.cuda:
                pytorch_prob = pytorch_blobs[output_layer].data.cpu().view(-1).numpy()
            else:
                pytorch_prob = pytorch_blobs[output_layer].data.view(-1).numpy()
            # caffe_prob = caffe_blobs[output_layer].data[0]
            print('all prob:')
            print(pytorch_prob)
            print('pytorch classification top1: %f %s' % (pytorch_prob.max(), synset_dict[pytorch_prob.argmax()]))

