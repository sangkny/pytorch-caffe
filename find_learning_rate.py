# -*- coding: utf-8 -*-
import sys
from caffenet import *
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

# -- parkingclassify
import numpy as np
import os
import argparse
from PIL import Image
import torchvision as tv

# train loader implementation from parkingclassify
class PK_train_data():
    def __init__(self, folder,transform = None, crop_size = 32):
        super(PK_train_data, self).__init__()
        file_0 = os.listdir(os.path.join(folder, '0'))
        file_1 = os.listdir(os.path.join(folder, '1'))
        self.labels = []
        self.fnames = []
        self.transform = transform
        #self.crop_size = crop_size

        for fileName in file_0:
            self.fnames.append(os.path.join(folder, '0',fileName))
            self.labels.append(0)
        for fileName in file_1:
            self.fnames.append(os.path.join(folder, '1',fileName))
            self.labels.append(1)
        self.num_samples = len(file_0) + len(file_1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target = self.pull_item(index)
        return img, target
    def pull_item(self, index):
        image_path = self.fnames[index]
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        if img.mode == 'L':
            img = img.convert('RGB')
        #img = img.convert('L') # gray scale
        img = img.convert('RGB')
        im_width, im_height = img.size # PIL image => size => width, height
        img = np.array(img)             # numpy => (h,w,c)
        #img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
            img = np.array(img)
        else:
            img = np.moveaxis(img, -1, 0)
        target = self.labels[index]
        img = img.astype('float32')
        return torch.from_numpy(img), target # convert to tensor, and its label which is tensor as well
    def getFiles(self): # return all the file names
        return self.fnames

def find_lr(net, criterion, optimizer, trn_loader, init_value=1e-8, final_value=10., beta=0.98, device='cuda:0'):
    net.to(device) # make net depending the device
    num_in_epoch = len(trn_loader) - 1
    mult = (final_value / init_value) ** (1 / num_in_epoch)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    displayfreq = 500
    for fileidx, data in enumerate(trn_loader):
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, labels = data # should be tensor
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device)) # convert to device-depent Variable
        optimizer.zero_grad()
        pred = net(inputs)
        loss = criterion(pred['fc_blob3'], labels) # by sangkny pred should give last output...
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item() # loss.data[0] by dangnky
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4000* best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(torch.log10(torch.FloatTensor([lr]).to(device)))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

        if fileidx % displayfreq == 0 and batch_num > 1:
            print("processing .... {} %".format(float(fileidx) * 100. / float(num_in_epoch)))

    return log_lrs, losses

def Caffe2PytorchNet(protofile, weightfile, ch_num = 1):
    net = CaffeNet(protofile, width=args.width, height=args.height, channels=ch_num, omit_data_layer=True, phase='TEST')
    if args.cuda:
        net.cuda()
    #print(net)
    net.load_weights(weightfile)
    net.eval()

    return net

'''
Example run. Select highest LR before graph looks erratic
model = BasicEmbModel(len(occupations), len(locations), len(industries), len(job_tags), len(user_ids), 25).to(device)
occu_warp = partial(warp_loss, num_labels=torch.FloatTensor([len(adids)]).to(device), device=device, limit_grad=False)
logs,losses = find_lr(model, occu_warp, torch.optim.RMSprop(model.parameters(), lr=0.05), train_loader, init_value=0.001, final_value=1000)
m = -5
plt.plot(logs[10:m],losses[10:m])
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Learning Rate Finder')
    parser.add_argument('--srcDir', default='./pk_data/20190812_all_images', type=str, help='base directory')
    parser.add_argument('--showImg', default=True, type=bool, help='flag to show images during process')

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
    parser.add_argument('--cuda', default=True, type=bool,
                        help='enables cuda')  # gives False which is faster than using GPU
    parser.add_argument('--batchFlag', default=True, type=int, help='batch processing Flag')

    args = parser.parse_args()
    # parameter settings
    args.srcDir = "D:/sangkny/pyTest/MLDL/NexQuadDataSets/3channels/40x32"  # 4 phase : camera input 40x32
    args.showImg = False
    args.saveImg = True

    output_layer = "fc_blob3"  # correct the proper output layer, default:'prob'

    protofile = args.protofile
    weightfile = args.weightfile

    model_height = args.height
    model_width = args.width
    model_channels = 3  # args.channels
    color_input = True if (model_channels == 3) else False
    describe_layer = False
    Net_display_once = True
    save_inc_files = True

    # trainloader
    image_crop_size = 32
    image_crop_h = 40  # 현재는 상관없음.. 싸이즈 조정 없음..
    image_crop_w = 32
    image_crop_size = image_crop_h if image_crop_h < image_crop_w else image_crop_w
    train_batch_size = 10  # 1024*2

    # define the transform for augmentation
    train_data_transforms = tv.transforms.Compose([
        tv.transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0),
        tv.transforms.ToTensor(),
    ])
    pkdata_train = PK_train_data(args.srcDir, train_data_transforms)
    train_pk_loader = torch.utils.data.DataLoader(dataset=pkdata_train,
                                                  batch_size=train_batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
    #load models
    # --  load Network and model from config and caffemodel files
    pytorch_net = Caffe2PytorchNet(protofile, weightfile, ch_num=model_channels)
    pytorch_net.set_verbose(False)  # display flow inside the network on/off
    if (args.cuda and torch.cuda.is_available()):
        net = torch.nn.DataParallel(pytorch_net)
    # --  load files done -----------------------------------

    # find learning rate
    initvalue = 0.001
    finalvalue = 10
    logs, losses = find_lr(pytorch_net, nn.CrossEntropyLoss(), torch.optim.SGD(pytorch_net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4), train_pk_loader,
                           init_value=initvalue, final_value=finalvalue, device='cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # display the result
    m = -5
    found_lr = 0;
    if len(logs) > 20:
        plt.plot(logs[10:m], losses[10:m])
        #found_lr = 10**(logs[10+np.argmin(losses[10:m])-len(logs[10:m])//8]) # why - ... // 8?
        found_lr = 10 ** (logs[10 + np.argmin(losses[10:m])])  # why - ... // 8?
        print('idx: %s, log loss: %s'%(np.argmin(losses[10:m]), logs[10+np.argmin(losses[10:m])]))

    else:
        plt.plot(logs[:], losses[:])
        #found_lr = 10 ** (logs[np.argmin(losses)-len(logs)//10])
        found_lr = 10 ** (logs[np.argmin(losses)])
        print('idx: %s, log loss: %s' % (np.argmin(losses[:]), logs[np.argmin(losses[:])]))
    plt.xlabel('learning rate (log scale)', labelpad=1)
    plt.ylabel('loss', labelpad=1)
    # put text in the figure
    txtstr = 'init:{%s}, max:{%s}, batch_size:{%s}, found_lr:{%.8f}\n'%(initvalue, finalvalue, train_batch_size,found_lr)
    plt.title(txtstr)
    plt.show()
    print('# of data: %s'%(len(logs)))
    print('Found lr: {}'.format(found_lr.data[0]))

