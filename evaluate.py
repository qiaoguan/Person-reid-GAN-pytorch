import scipy.io
import torch
import numpy as np
import time

#######################################################################
# Evaluate     find the corresponding person in the gallery set
# qf: 1*1024, ql:1, qc:1,   gf: 19732*1024, gl: 19732, gc: 19732
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf
    score = np.dot(gf,query)     # cosine distance
    # predict index
    index = np.argsort(score)  #from small to large(small equals similar)
    index = index[::-1]   # reverse    19732*1
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)  # 59*1
    camera_index = np.argwhere(gc==qc) # 3156*1
    
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)  # the difference of the set    51*1
    junk_index1 = np.argwhere(gl==-1)                                                                      # 3819*1
    junk_index2 = np.intersect1d(query_index, camera_index)  # find the intersection of two arrays         # 8*1
    junk_index = np.append(junk_index2, junk_index1) #.flatten())                                          # 3827*1
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()   # create a zero tensor
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)  
    index = index[mask]    # remove junk_index from index

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)  # np.argwhere,  find the index of element which satisfy the condition that mask is True
    rows_good = rows_good.flatten()
    '''
    print(index.shape)
    print(ngood)
    print(rows_good.shape)
    name=raw_input()
    '''
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = result['query_f']    # 3368*1024
query_cam = result['query_cam'][0]   # 3368
query_label = result['query_label'][0]  # 3368
gallery_feature = result['gallery_f']   #19732*1024
gallery_cam = result['gallery_cam'][0]  #19732
gallery_label = result['gallery_label'][0]  # 19732

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)   len=3368
'''
print('---------------------')
print(len(query_feature))
print(len(query_label))
print(len(query_cam))
print(len(gallery_feature))
print(len(gallery_label))
print(len(gallery_cam))
'''
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
  #  print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))