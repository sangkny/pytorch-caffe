# -*- coding: utf-8 -*-
# 2020. 02. 26 sangkny : save incorrect files to a specific folder
# 2020. 02. 24 sangkny for batch processing
# 2020. 02. 12 sangkny for verifying the Nexquad Lenet performance
import sys
from caffenet import *
import numpy as np
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time
import os
import shutil # for copy2


def load_image(imgfile, color_input):
    input_color = color_input
    image = caffe.io.load_image(imgfile, color=input_color)
    # interpolating the image as below
    if input_color:
        transformer = caffe.io.Transformer(
            {'data': (1, 3, args.height, args.width)})  # sangkny was (1,3, args.height, args.width) is color image

        transformer.set_transpose('data', (2, 0, 1)) # channel first
        transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR])) # mean extraction
        # transformer.set_mean('data', np.array([128]))
        transformer.set_raw_scale('data', args.scale) # scale
        transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR
        image = transformer.preprocess('data', image)
        image = image.reshape(1, 3, args.height, args.width) # channel extension for deep learning model input

    else:
        transformer = caffe.io.Transformer(
            {'data': (1, 1, args.height, args.width)})  # sangkny was (1,3, args.height, args.width) is color image
        # transformer.set_transpose('data', (2, 0, 1)) # not required for gray
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

def Caffe2PytorchNet(protofile, weightfile, ch_num = 1):
    net = CaffeNet(protofile, width=args.width, height=args.height, channels=ch_num, omit_data_layer=True, phase='TEST')
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
    parser = argparse.ArgumentParser(description='Inference for caffe model with pytorch')
    parser.add_argument('--protofile', default='', type=str)
    parser.add_argument('--weightfile', default='', type=str)
    parser.add_argument('--imgfile', default='', type=str)
    parser.add_argument('--height', default=224, type=int)
    parser.add_argument('--channels', default=1, type=int, help='number of channels (default: 1)')
    parser.add_argument('--width', default=224, type=int)
    parser.add_argument('--meanB', default=104, type=float)
    parser.add_argument('--meanG', default=117, type=float)
    parser.add_argument('--meanR', default=123, type=float)
    parser.add_argument('--scale', default=255, type=float)
    parser.add_argument('--synset_words', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')    # gives False which is faster than using GPU
    parser.add_argument('--batchFlag', default=True, type=int, help='batch processing Flag')
    parser.add_argument('--batchDir', default='./data', type=str, help='select a folder for batch process')
    parser.add_argument('--conf_th', default=0.8, type=float, help='confidence threshold')

    args = parser.parse_args()
    print(args)
    # ---- parameter settings ---------------
    output_layer = "fc_blob3" # correct the proper output layer, default:'prob'

    protofile = args.protofile
    weightfile = args.weightfile
    #imgfile =  'C:\\Users\\mmc\\Downloads\\20200214_test_result\\1\\vpdImage_19701010_132301_7.jpg' #args.imgfile
    #imgfile = 'C:\\Users\\mmc\\Downloads\\Train.Haar\\side\\left\\Left_SG\\L_sg_00085.bmp'
    #imgfile = './data/cat.jpg'
    batchFlag = args.batchFlag  #batch flag
    #batchDir = 'C:\\Users\\mmc\\Downloads\\Train.Haar\\side\\left\\Left_SG'  # args.batchDir    # batch directory

    # --- Set the following information to test the model performance after changing the model in the args model info --
    batchDir = 'D:\\sangkny\\pyTest\\MLDL\\NexQuadDataSets\\3channels\\40x32\\0' #args.batchDir    # batch directory
    #batchDir = 'D:\\sangkny\\pyTest\\MLDL\\NexQuadDataSets\\3channels\\haar\\1'  # args.batchDir    # batch directory
    #batchDir = 'E:\\nexquad-ralated\\5cameras\\gather_images\\jpgs\\20200826_3chs_data_br04_ver\\train\\0'  # args.batchDir
    #E:\nexquad-ralated\5cameras\20200807\lenet40x32_20200729_4phase_20200804_171435_1440x1080
    #'D:/sangkny/pyTest/MLDL/NexQuadDataSets/3channels/40x32-ext/0'
    #batchDir = 'E:/nexquad-ralated/5cameras/gather_images/jpgs/3channels/vpd_images_model_v20200916_20200919_202553/0'  # args.batchDir
    #batchDir = 'D:/sangkny/pyTest/MLDL/NexQuadDataSets/3channels/40x32/0'  # args.batchDir
    #batchDir = 'D:\\sangkny\\pyTest\MLDL\\codes\\parkingClassify-master\\augimg_20200812_3channels_br04\\1' #args.batchDir    # batch directory

    #batchDir = 'C:\\Users\\mmc\\Downloads\\new_sample\\nobj'  # args.batchDir    # batch directory
    gt_class = os.path.split(batchDir)[-1]  # ground truth class

    #incFileDir = 'D:\\sangkny\\pyTest\\MLDL\\NexQuadDataSets\\4phase\\19-20\\incorrect_20200629_v3_all_data_60000' # incorrect folder base
    #incFileDir = 'E:\\nexquad-ralated\\5cameras\\gather_images\\jpgs\\20200826_3chs_data_br04_ver\\train\\incorrect_20200918_lr00001_4_3chs_b512_br04_High_30000'  # incorrect folder base
    #incFileDir = 'E:/nexquad-ralated/5cameras/gather_images/jpgs/3channels/vpd_images_model_v20200916_20200919_202553/incorrect_20200918_lr00001_4_3chs_b512_br04_SVG_200000'  # incorrect folder base
    #incFileDir = 'E:\\nexquad-ralated\\5cameras\\gather_images\\jpgs\\gather_dataset_20200608\\incorrect_40000'  # incorrect folder base
    incFileDir = 'D:/sangkny/pyTest/MLDL/NexQuadDataSets/3channels/40x32/incorrect_20200920_lr00001_4_3chs_b512_br04_SVG_185000'  # incorrect folder base
    #incFileDir = 'D:/sangkny/pyTest/MLDL/NexQuadDataSets/3channels/haar/incorrect_20200916_lr00001_4_3chs_b512_br04_High_200000'  # incorrect folder base
    model_height = args.height
    model_width = args.width
    model_channels = 3 # args.channels
    color_input = True if(model_channels==3) else False
    describe_layer = False
    Net_display_once = True
    save_inc_files = True
    save_inc_files_with_conf = True # file name extension with confidence level
    save_inc_files_All = False # save all regardless of confidence threshold
    debug_text = False
    confidence_threshold = args.conf_th



    # --  load Network and model from config and caffemodel files
    # time_pytorch, pytorch_blobs, pytorch_models, pytorch_net = forward_pytorch(protofile, weightfile, image)
    pytorch_net = Caffe2PytorchNet(protofile, weightfile, ch_num=model_channels)
    pytorch_net.set_verbose(False)                          # display flow inside the network on/off
    pytorch_models = pytorch_net.models
    layer_names = pytorch_models.keys()
    if args.synset_words != '':
        synset_dict = load_synset_words(args.synset_words)
    # --  load files done -----------------------------------

    # --------- variables --------------------------------------
    listOfFiles = list()
    refinedFiles = list()
    outputs =[]
    total_times = list()
    inc_Files = list()  # incorrect files
    inc_Probes = list() # incorrect file probabilities
    displayfreq = 1000
    # -------------- variables end ---------------------------

    # --  batch process or single process
    if batchFlag and os.path.isdir(batchDir):   # batch image processing
        #search around
        print('\n Searching files under %s ... \n' % batchDir)
        for (dirpath, dirnames, filenames) in os.walk(batchDir):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        for idx, file in enumerate(listOfFiles):  # dir and its idx
            ext = str(file).lower()
            if (("bmp" in ext) or ("jpg" in ext) or ("png" in ext)):
                refinedFiles.append(file)

    else:                               # single image processing
        refinedFiles.append(imgfile)

    allfiles = len(refinedFiles)  # all files to be processed

    # processing the given file(s)
    for fileidx, file in enumerate(refinedFiles):
        image = load_image(file, color_input)
        # convert the image according to the proper input for protofile
        # 1 channel
        # convert already bgr in previous step
        # gray = np.dot(image[...,:3],[0.114, 0.587, 0.299])
        if(color_input==False): # or model_channels =1
            image = image[0, 0, :, :]

            # --------------------show the given image ------------
            import cv2
            cv2.imshow('test', np.array(image/255))
            cv2.waitKey(1)
            # ----------------------------------
            image = image.reshape(1, 1, model_height, model_width)
        else:# image should have 3 channels and its shape(1,3,40,32)
            if(debug_text):
                print(image.shape)

        image = torch.from_numpy(image)
        if args.cuda:
            image = Variable(image.cuda())
        else:
            image = Variable(image)
        t0 = time.time()
        pytorch_blobs = pytorch_net(image)    # forward step : update blobs
        t1 = time.time()
        time_pytorch = t1-t0
        total_times.append(time_pytorch)

        if(Net_display_once):
            print('\n pytorch net \n')
            print(pytorch_net)
            Net_display_once = False
        if (debug_text):
            print('pytorch forward time for the given image: %f msec' % (time_pytorch*1000))
            print('------------ Classification ------------')

        pytorch_blobs = pytorch_net.blobs
        blob_names = pytorch_blobs.keys()
        if output_layer in blob_names:
            if args.cuda:
                pytorch_prob = pytorch_blobs[output_layer].data.cpu().view(-1).numpy()
            else:
                pytorch_prob = pytorch_blobs[output_layer].data.view(-1).numpy()
            # caffe_prob = caffe_blobs[output_layer].data[0]
            if(debug_text):
                print("file:{}".format(file))
                print('all (softmax(sigmoid)) prob-> confidence : {}'.format(
                    torch.softmax(torch.sigmoid(pytorch_blobs[output_layer].data.view(-1)/255), dim=0)))  # softmax after sigmoid for normalized value
                print('softmax prob -> confidence : {}'.format(F.softmax(pytorch_blobs[output_layer].data.view(-1)/255, dim=0))) # softmax for normalized value
                if np.max(F.softmax(pytorch_blobs[output_layer].data.view(-1)/255, dim=0).numpy()) < 0.75:
                    print('-------------------- confidence is too low ------------------------------------------{}'.format(torch.softmax(pytorch_blobs[output_layer].data.view(-1)/255, dim=0)))
                print('final fc data: {}'.format(pytorch_prob))
                print('pytorch classification top1: %f %s' % (pytorch_prob.max(), synset_dict[pytorch_prob.argmax()]))
            else:
                if fileidx % displayfreq == 0:
                    print("processing .... {} %".format(float(fileidx)*100./len(refinedFiles)))

            outputs.append(pytorch_prob.argmax())
            if(save_inc_files and (int(pytorch_prob.argmax()) is not int(gt_class))):
                # save incorrect files
                inc_Files.append(file)
                inc_Probes.append(np.max(F.softmax(pytorch_blobs[output_layer].data.view(-1)/255, dim=0).numpy()))

    # network inference ends
    avg_time = np.sum(total_times)/allfiles
    print('avg time for {} images: {} msec'.format(allfiles, avg_time*1000))
    print('\n outputs:{} \n sum: {}\n acc: {}% \n'.format(outputs, np.array(outputs).sum(),np.array(outputs).sum()/allfiles*100))
    # save incorrect files for analysis
    if save_inc_files:
        newFilesDir = incFileDir
        if not "incorrect" in newFilesDir:
            newFilesDir = os.path.join(newFilesDir, "incorrect")
        newFilesDir = os.path.join(newFilesDir, str(gt_class))
        if not os.path.isdir(newFilesDir):
            os.makedirs(newFilesDir)
            #create all intermediate folders : os.makedirs(path), a single folder: os.mkdir(path)

        print(' *&* writing incorrect files to {} ... \n'.format(newFilesDir))
        assert(len(inc_Files)==len(inc_Probes))
        skip_count = 0;
        for _idx, _file in enumerate(inc_Files):
            if confidence_threshold <= round(float(inc_Probes[_idx]),5) or save_inc_files_All:
                print('{}: {} --> prob: {}\n'.format(_idx, _file, inc_Probes[_idx]))
                if save_inc_files_with_conf: # save with confidence level
                    des_file_name = os.path.split(_file)[-1]                        # full file name
                    des_file_name_only = des_file_name[:des_file_name.rfind('.')]   # file name without ext
                    des_file_ext = des_file_name[des_file_name.rfind('.')+1:]       # extension only
                    _dfile = os.path.join(newFilesDir, des_file_name_only+'-'+'%.4f'%(inc_Probes[_idx])+'.'+des_file_ext) # or _file.split(os.path.sep)[-1]
                else:
                    _dfile = os.path.join(newFilesDir, os.path.split(_file)[-1]) # or _file.split(os.path.sep)[-1]
                shutil.copy2(_file, _dfile)
            else:
                skip_count +=1
        print('==>> skipped incorrect files due to confidence th:{} --> skip_count:{}'.format(confidence_threshold, skip_count))
    # ---------------------------- display parameters -----------------------
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
        #print('------------ Output Difference ------------')
