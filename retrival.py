import sys
sys.path.insert(0,'/core1/data/home/liuhuawei/evalution_ai/utils/')
sys.path.insert(0,'/home/shenyaxin/pre/src/retrival')
import json
import cv2
import requests
import numpy as np
from sklearn.preprocessing import normalize
from PIL import Image
from matplotlib import pyplot as plt
from knn_speed import KNNSPEED
import time
sys.path.insert(0,'/core1/data/home/xuqiang/mysql/')
from mysql import MysqlOperator
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
    
def get_fea(file_name):
    data = json.load(open(file_name, 'r'))
    fea = data['fea']
    src_id = data['src_id']
    fea = np.array(fea, dtype = 'float32')
    fea = normalize(fea)
    return fea, src_id
    
def get_fea_hsv(file_name):
    data = json.load(open(file_name, 'r'))
    fea = data['fea']
    src_id = data['src_id']
    fea = np.array(fea, dtype = 'float32')
    sv = fea[:,50:]
    #sv = fea
    sv = normalize(sv)
    return sv, src_id

def show_img(mysql, src_ids):
    imgs = []
    for src_id in src_ids:
        sql = '''select concat('http://192.168.1.23:8082/', img.fid) as url
             from internal_website.image img
             where img.src_id = {}'''.format(src_id)
    
        img_name = list(mysql.select(sql))[0]['url']
        #print img_name
        if img_name[:4] == 'http':
            type_ = 'url'
        else:
            type_ = 'path'
        img = cv_load_image(img_name, type_)[:,:,(2,1,0)]
        img = cv2.resize(img, (256,256))
        imgs.append(img)
    #plt.subplot(3,4,1)
    for i in range(8):
        plt.subplot(3,4,i+5)
        plt.title('top %d'%(i))
        plt.imshow(imgs[i+1])
        plt.axis('off')
    plt.subplot(3,4,1)
    plt.title('query')
    plt.imshow(imgs[0])
    plt.axis('off')
    plt.subplot(3,4,2)
    plt.title('R')
    plt.imshow(imgs[0][:,:,0], plt.cm.gray)
    plt.axis('off')
    
    plt.subplot(3,4,3)
    plt.title('G')
    plt.imshow(imgs[0][:,:,1], plt.cm.gray)
    plt.axis('off')
    
    plt.subplot(3,4,4)
    plt.title('B')
    plt.imshow(imgs[0][:,:,2], plt.cm.gray)
    plt.axis('off')
    
    plt.show()

if __name__=='__main__':
    mysql = MysqlOperator()
    fea_q, id_q = get_fea_hsv('/data/data/shenyaxin/fea/color_hsv_hist_1.json')
    fea_db, id_db = get_fea_hsv('/data/data/shenyaxin/fea/color_hsv_hist_0.json')
    fea_q2, id_q2 = get_fea('/data/data/shenyaxin/fea/color_fea_27_1.json')
    fea_db2, id_db2 = get_fea('/data/data/shenyaxin/fea/color_fea_27_0.json')
    fea_q3, _ = get_fea('/data/data/shenyaxin/fea/res101_fea_iter18W_1.json')
    fea_db3, _ = get_fea('/data/data/shenyaxin/fea/res101_fea_iter18W_0.json')
    
    fea_q = np.concatenate((0.1*fea_q2, 0.3*fea_q, fea_q3), axis = 1)
    fea_db = np.concatenate((0.1*fea_db2, 0.3*fea_db, fea_db3), axis = 1)
    knn = KNNSPEED(features = fea_db, titles = id_db)
    #id_q = id_q[3000:]
    #fea_q = fea_q[3000:]
    re = np.zeros(7)
    beg = time.time()
    k_cand = [1,5,10,15,20,25,30]
    for idx, id in enumerate(id_q):
        res = knn.tops(fea = fea_q[idx], k = 50)
        res_id = [item[0] for item in res]
        #res_id = res[:,0]
        for ind, kk in enumerate(k_cand):
            if id in res_id[:kk]:
                re[ind] += 1
        plot_id = [id]
        plot_id.extend(res_id[:9])
        show_img(mysql, plot_id)
        if idx % 1000 == 0:
            print '---------------{} imgs done----------------'.format(idx)
            print re
            print time.time()-beg
            beg = time.time()
            
    retrival = re/len(id_q)
    print re
    print retrival