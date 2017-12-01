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

data_c1 = json.load(open('/data/data/shenyaxin/fea/color_fea_1.json', 'r'))
data_c0 = json.load(open('/data/data/shenyaxin/fea/color_fea_0.json', 'r'))
data_r1 = json.load(open('/data/data/shenyaxin/fea/res101_fea_iter18W_1.json', 'r'))
data_r0 = json.load(open('/data/data/shenyaxin/fea/res101_fea_iter18W_0.json.json', 'r'))
fea_c1 = data_c1['fea']
src_c1 = data_c1['src_id']
fea_c0 = data_c0['fea']
src_c0 = data_c0['src_id']
fea_r1 = data_r1['fea']
src_r1 = data_r1['src_id']
fea_r0 = data_r0['fea']
src_r0 = data_r0['src_id']

fea_all_0 = 
fea_all_1 = 
