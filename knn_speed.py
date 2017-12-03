import numpy as np
from sklearn.preprocessing import normalize
import sys
import time
from memory_profiler import profile
import gc

sys.path.insert(0, '/core/data/deploy/server_image/faiss')
import faiss

class KNNSPEED(object):
    '''
    KNN class for finding most "similar" items of a query item.
    '''
    def __init__(self, features=None, titles=None, data=None, need_norm=True):
        '''
        init the dataset. Specify the ids of items' and corresponding features separately or by a dict.
        Args:
            features(list(float)): features of items to search.
            titles(list(str)): ids of items to search. In the same sequence with the features.
            data(dict): ids and features stored in data. Id is key and feature is value. If data is not None, features and titles are ignored.

        Example:
            KNN.init([[1.,1.,1.], [1.,2.,2.]],['x','y'])
        '''        
        if data is None and features is None:
            raise
        #tmp = data.values()
        features = np.array(tmp, dtype=np.float32) if data is not None else np.array(features, dtype=np.float32)
        #del tmp
        gc.collect()
        self.title = list(data.keys()) if data is not None else titles
        print 'Db feature shape: ', features.shape
        self.index = faiss.IndexFlatL2(features.shape[1])
        if need_norm == True:
            self.index.add(normalize(features, copy=False))
        else:
            self.index.add(features)
#        del features
       # del data
        gc.collect()

    def tops(self, fea, k=5, candidates=None, need_norm=True):
        '''
        Get top k most similar items from stock. If specified candidates, only get top k from the candidates.
        Args:
            fea(list/numpy.ndarray): feature of query image.
            k(int): top k most similar items.
            candidates(dict/list): the candidate item ids.

        Returns:
            topk(list): return top k (itemid, score) pair.

        Example:
            KNN.tops()
        '''
        
        beg_time = time.time()
        # fea = fea[np.newaxis, :] if fea.ndim==1 else fea
        fea = fea.reshape(1, -1) if fea.ndim==1 else fea
        if candidates is None:
            if need_norm:
                D, I = self.index.search(normalize(fea), k)
                # D, I = self.index.search(np.ascontiguousarray(normalize(fea)), k)
            else:
                D, I = self.index.search(fea, k)    
            res = [(self.title[i[0]], i[1]) for i in zip(I[0], D[0])]
#        print >> sys.stderr, 'Timing knn', time.time() - beg_time    
        return res    



