# -*- coding: utf-8 -*-
import numpy as np
import sys, os
import types
import cv2
import caffe
import requests
from skimage import io
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
        img = cv2.imread(in_, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return img
    
def transform_img(img_name, bbox):
    if img_name[:4] == 'http':
        type_ = 'url'
    else:
        type_ = 'path'
    img = cv_load_image(img_name, type_).astype(np.float32)
    # print bbox
    # print len(bbox)
    # raise
    bbox = np.array(bbox).reshape((1, -1))
    bbox = scale_bbox(bbox, 1.1).astype(int)[0]
    H, W, _ = img.shape
    bbox[2] = min(bbox[2], W)
    bbox[3] = min(bbox[3], H)
    img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    img = cv2.resize(img, (224, 224))
    
    mean = np.array([104.0, 117.0, 123.0], dtype=np.float32).reshape((1, 1, -1))
    img -= mean
    img = img.transpose((2, 0, 1))
    return img

def get_data(mysql, src):
    sql = '''
        select test.src_id src_id, obj.x_pixel xmin, obj.y_pixel ymin, obj.width_pixel width, obj.height_pixel height,
        concat('http://192.168.1.23:8082/', img.fid) as url
        from internal_website.makeup_comment2shop test
        inner join internal_website.image img 
        on img.id = test.img_id
        inner join internal_website.object obj
        on obj.img_id = img.id
        where test.src_type = {}
        '''.format(src)
    datas = list(mysql.select(sql))
    print len(datas)
    return datas

def vis_square(data):
    #data = (data - data.min()) / (data.max() - data.min())

    # 强制性地使输入的图像个数为平方数，不足平方数时，手动添加几幅
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # 每幅小图像之间加入小空隙
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
                           + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)   

    # 将所有输入的data图像平复在一个ndarray-data中（tile the filters into an image）
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        # data的一个小例子,e.g., (3,120,120)
        # 即，这里的data是一个2d 或者 3d 的ndarray
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        # 显示data所对应的图像
    plt.imshow(data); plt.axis('off')
    plt.show()
        
if __name__ == '__main__':
    caffe.set_device(1)
    caffe.set_mode_gpu()

    net_file = '/home/shenyaxin/color/deploy.prototxt'
    caffe_model = '/core1/data/home/liuhuawei/tools/models/bvlc_reference_caffenet.caffemodel'
    net = caffe.Net(net_file, caffe_model, caffe.TEST)
    mysql = MysqlOperator()
    #shop_mete = get_data(mysql, 0)
    datas = get_data(mysql, 1)
    src_id_list = []
    fea_list = []
    batch_size = 1
    bs = 1
    for i in range(0, len(datas), batch_size):
        if (i+batch_size)>len(datas):
            bs = len(datas)-i
        data_batch = datas[i:i+bs]
        imgs = []
        itemids = []
        for data in data_batch:
            itemid = data['src_id']
            itemids.append(itemid)
            path = data['url']
            xmin = int(data['xmin'])
            ymin = int(data['ymin'])
            width = int(data['width'])
            height = int(data['height'])
            bbox = [xmin, ymin, xmin+width, ymin+height]
            img = transform_img(path, bbox)
            imgs.append(img)
        
        #transformerd = caffe.io.Transformer({'test_photod': net.blobs['data'].data.shape})
        #transformerd.set_transpose('test_photod', (2, 0, 1))
        #transformerd.set_mean('test_photod', np.array([104,117,123]))
        imgs = np.array(imgs)
        net.blobs["data"].reshape(bs,3,224,224)
        net.blobs["data"].data[...] =  imgs
        output = net.forward()
        '''fea = net.blobs['pool1'].data.copy()
        print fea.shape
        fea = fea[:,48:,:,:]
        print fea.shape
        #exit()
        fea = fea.reshape(bs, -1)
        print fea.shape
        exit()
        src_id_list.extend(itemids)
        fea_list.extend(fea.tolist())
        curlen = len(fea_list)
        if curlen%100 == 0:
            print 'now: ', curlen
        '''
        w1 = net.params["conv1"][0].data
        print w1.shape
        vis_square(w1.transpose(0,2,3,1))
        fea = net.blobs['conv1'].data.copy()
        print fea.shape
        
        vis_square(fea.transpose(1,0,2,3))
        
   