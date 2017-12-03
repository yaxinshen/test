# -*- coding: utf-8 -*-
import numpy as np
import sys, os
import types
import cv2
import caffe
import requests
from skimage import io,data,color
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
caffe_path = '/core1/data/home/liuhuawei/detection/caffe/python/'
sys.path.insert(0, caffe_path)
from caffe.proto import caffe_pb2
sys.path.insert(0,'/core1/data/home/liuhuawei/evalution_ai/utils/')
from data_augmentation import scale_bbox
sys.path.insert(0,'/core1/data/home/xuqiang/mysql/')
from mysql import MysqlOperator
import json

cv_session = requests.Session()
cv_session.trust_env = False

def cv_load_image(in_, type_='path'):
    '''
    Return
        image: opencv format np.array. (C x H x W) in BGR np.uint8
    '''
    if type_ == 'url':
        img_nparr = np.fromstring(cv_session.get(in_).content, np.uint8)
        img = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(in_, cv2.IMREAD_COLOR)
    return img
    
def transform_img(img_name, bbox):
    if img_name[:4] == 'http':
        type_ = 'url'
    else:
        type_ = 'path'
    img = cv_load_image(img_name, type_)#.astype(np.float32)
    #plt.subplot(236), plt.imshow(img[:,:,(2,1,0)])
    # print bbox
    # print len(bbox)
    # raise
    bbox = np.array(bbox).reshape((1, -1))
    bbox = scale_bbox(bbox, 1.1).astype(int)[0]
    H, W, _ = img.shape
    bbox[2] = min(bbox[2], W)
    bbox[3] = min(bbox[3], H)
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    #plt.subplot(234), plt.imshow(img)
    #plt.subplot(235), plt.imshow(img[:,:,(2,1,0)])
    #img = cv2.resize(img, (224, 224))
    
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #hsv = img
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #H, S, V = cv2.split(HSV)
    #print hsv
    #print hsv.shape
    
    
    hist = [cv2.calcHist([hsv], [0], None, [50], [0, 180]), \
    cv2.calcHist([hsv], [1], None, [50], [0, 256]), \
    cv2.calcHist([hsv], [2], None, [50], [0, 256])]
    hist = np.array(hist)
    x = np.arange(50) + 0.5
    #plt.subplot(231), plt.bar(x, hist[0])
    #plt.subplot(232), plt.bar(x, hist[1])
    #plt.subplot(233), plt.bar(x, hist[2])
    #plt.bar(x, hist[0])
    #plt.savefig('img/hist.jpg')
    #hsv = hsv.transpose((2, 0, 1))
    #mean = np.array([104.0, 117.0, 123.0], dtype=np.float32).reshape((1, 1, -1))
    #img -= mean
    #img = img.transpose((2, 0, 1))
    return img,hsv,hist

def get_data(mysql, src):
    sql = '''
        select obj.x_pixel xmin, obj.y_pixel ymin, obj.width_pixel width, obj.height_pixel height,
        img.path as path, img.id as img_id
        from dp_image.image img
        inner join dp_image.object obj
        on obj.img_id = img.id
        where img.src_src = 'dp_customer2shop' and img.split_type = 'test' limit 5000
        '''

    datas = list(mysql.select(sql))
    print len(datas)
    return datas

def vis_square(data):
    data = (data - data.min()) / (data.max() - data.min())

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
                           + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)   

    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data); plt.axis('off')
    plt.show()
        
if __name__ == '__main__':
    #caffe.set_device(0)
    #caffe.set_mode_gpu()

    #net_file = '/home/shenyaxin/color/deploy.prototxt'
    #caffe_model = '/core1/data/home/liuhuawei/tools/models/bvlc_reference_caffenet.caffemodel'
    #net = caffe.Net(net_file, caffe_model, caffe.TEST)
    mysql = MysqlOperator()
    #shop_mete = get_data(mysql, 0)
    datas = get_data(mysql,0)
    #src_id_list = []
    fea_list = []
    img_id_list = []
    batch_size = 32
    bs = 32
    for i in range(0, len(datas), batch_size):
        if (i+batch_size)>len(datas):
            bs = len(datas)-i
        data_batch = datas[i:i+bs]
        imgs = []
        hsvs = []
        hists = []
        img_ids = []
        for data in data_batch:
            img_id = data['img_id']
            img_ids.append(img_id)
            path = data['path']
            xmin = int(data['xmin'])
            ymin = int(data['ymin'])
            width = int(data['width'])
            height = int(data['height'])
            bbox = [xmin, ymin, xmin+width, ymin+height]
            _,_,hist = transform_img(path, bbox)
            #print hist[0].max(),hist[1].max(),hist[2].max()
            #exit()
            hists.append(hist)
            #imgs.append(img)
            #hsvs.append(hsv)
            #exit()
        #transformerd = caffe.io.Transformer({'test_photod': net.blobs['data'].data.shape})
        #transformerd.set_transpose('test_photod', (2, 0, 1))
        #transformerd.set_mean('test_photod', np.array([104,117,123]))
        '''imgs = np.array(imgs)
        net.blobs["data"].reshape(bs,3,224,224)
        net.blobs["data"].data[...] =  hsvs
        output = net.forward()
        fea = net.blobs['poolhsv'].data.copy()'''
        #fea = fea[:,48:,:,:]
        #print fea.shape
        fea = np.array(hists)
        fea = fea.reshape(bs, -1)
        #print fea.shape
        #exit()
        #src_id_list.extend(itemids)
        img_id_list.extend(img_ids)
        fea_list.extend(fea.tolist())
        curlen = len(fea_list)
        if curlen%100 == 0:
            print 'now: ', curlen
            
        
        #w1 = net.params["conv1"][0].data
        #print w1.shape
        #vis_square(w1.transpose(0,2,3,1))
        
    json.dump({'fea':fea_list, 'id':img_id_list}, \
        open('/data/data/shenyaxin/fea/color/cloth_hsv_hist_{}.json'.format(1), 'w'))
    print len(img_id_list)
