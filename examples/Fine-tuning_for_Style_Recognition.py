#-*- coding:utf-8 –*-
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

import numpy as np
from pylab import *
import tempfile

#for unwhiten
# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

#for net
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

#######################################
#
#1. load weight file and label file
#
########################################
full_dataset = False
if full_dataset:
    NUM_STYLE_IMAGES = NUM_STYLE_LABELS = -1
else:
    NUM_STYLE_IMAGES = 2000
    NUM_STYLE_LABELS = 5

import os
weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
assert os.path.exists(weights)

# Load ImageNet labels to imagenet_labels
imagenet_label_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))#['n01440764 tench, Tinca tinca', 'n01443537 goldfish, Carassius auratus', ....]
assert len(imagenet_labels) == 1000
print 'Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...'])

# Load style labels to style_labels
style_label_file = caffe_root + 'examples/finetune_flickr_style/style_names.txt'
style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))#['Detailed', 'Pastel', 'Melancholy', 'Noir', 'HDR', 'Vintage', 'Long Exposure', 'Horror', 'Sunny', 'Bright', 'Hazy', 'Bokeh', 'Serene', 'Texture', 'Ethereal', 'Macro', 'Depth of Field', 'Geometric Composition', 'Minimal', 'Romantic']
if NUM_STYLE_LABELS > 0:
    style_labels = style_labels[:NUM_STYLE_LABELS]
print '\nLoaded style labels:\n', ', '.join(style_labels)




########################################
#
# Defining and running the nets
#
########################################
from caffe import layers as L
from caffe import params as P

#set learning rate mult
weight_param = dict(lr_mult=1, decay_mult=1)#{'lr_mult': 1, 'decay_mult': 1}
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2#[{'lr_mult': 0}, {'lr_mult': 0}]


#set classification net structor for imagenet and generate .prototxt
#no data layer
def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)#Local Response Normalization 0.1% performance
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name

#load caffenet
#利用DummyData设置备选图框，给新来的图像留个坑
dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))#shape:{'dim': [1, 3, 227, 227]} #DummyDataLayer fills any number of arbitrarily shaped blobs with random
imagenet_net_filename = caffenet(data=dummy_data, train=False)#构建deploy.prototxt
imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)


##################################################
# Define a function style_net which calls caffenet on data from the Flickr style dataset.
# 
# The new network will also have the CaffeNet architecture, with differences in the input and output:
# 
# the input is the Flickr style data we downloaded, provided by an ImageData layer
# the output is a distribution over 20 classes rather than the original 1000 ImageNet classes
# the classification layer is renamed from fc8 to fc8_flickr to tell Caffe not to load the original classifier (fc8) weights from the ImageNet-pretrained model
####################################################
def style_net(train=True, learn_all=False, subset=None):
    if subset is None:
        subset = 'train' if train else 'test'
    source = caffe_root + 'data/flickr_style/%s.txt' % subset
    transform_param = dict(mirror=train, crop_size=227, #layer类含有transform_param see http://caffe.berkeleyvision.org/tutorial/data.html
                           mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(transform_param=transform_param, source=source,#see http://caffe.berkeleyvision.org/tutorial/layers/imagedata.html
                                          batch_size=50, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)


#######################
#Call forward on untrained_style_net to get a batch of style training data.
#######################

untrained_style_net = caffe.Net(style_net(train=False, subset='train'),
                                weights, caffe.TEST)
untrained_style_net.forward()
style_data_batch = untrained_style_net.blobs['data'].data.copy()
style_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.int32)


#Pick one of the style net training images from the batch of 50 (we'll arbitrarily choose #8 here). Display it, then run it through imagenet_net, the ImageNet-pretrained network to view its top 5 predicted classes from the 1000 ImageNet classes.
def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]#start 从哪一层layer开始，有start，end参数 see http://blog.csdn.net/forest_world/article/details/52994117
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))

def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_style_preds(net, image):
    disp_preds(net, image, style_labels, name='style')

batch_index = 8
image = style_data_batch[batch_index]
matplotlib.pyplot.imshow(deprocess_net_image(image))
matplotlib.pyplot.savefig("filename.png")
matplotlib.pyplot.show()
print 'actual label =', style_labels[style_label_batch[batch_index]]

disp_imagenet_preds(imagenet_net, image)

disp_style_preds(untrained_style_net, image)


#We can also verify that the activations in layer fc7 immediately before the classification layer are the same as (or very close to) those in the ImageNet-pretrained model, since both models are using the same pretrained weights in the conv1 through fc7 layers.
diff = untrained_style_net.blobs['fc7'].data[0] - imagenet_net.blobs['fc7'].data[0]
error = (diff ** 2).sum()
assert error < 1e-8



