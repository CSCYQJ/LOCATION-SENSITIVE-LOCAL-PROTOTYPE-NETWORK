import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nb
import cv2 as cv
import torch
from models.fewshot_grid import FewShotSeg
import random
import torch.nn as nn
import numpy as np
import math

def get_range(volume,label):
    label_map=(volume==label).astype(np.uint8)
    _,_,batch = label_map.shape
    slice_with_class = np.sum(label_map.reshape(-1,batch), axis=0) > 200
    range_index=[]
    for i in range(batch-1):
        if slice_with_class[i]==True:
            range_index.append(i)
    return range_index

def scale_and_normalize(img_gray):
    if len(img_gray.shape)==2:
        rows,cols = img_gray.shape
    else:
        print('Wrong!')
        return img_gray
    im1 = img_gray.astype(float)
    A,B=im1.min(),im1.max()
    im1 -= A
    im1 /= B-A
    return im1

def cal_dice(seg1,seg2):
    return (2*np.sum(seg1*seg2)+0.001)/(np.sum(seg1+seg2)+0.001)

def load_vol_and_mask(volume_list,volumes_path,segs_path,labels=[1,2,13,14,17,18],vol_save_path="/home/qinji/PANet_results/vol/"):
    vol_dict={}
    seg_dict={}
    for vol in volume_list:
        print(vol)
        vol_path=os.path.join(volumes_path,vol)
        vol_nifty = nb.load(vol_path)
        vol_data = vol_nifty.get_fdata()
        seg_path=os.path.join(segs_path,vol.split('_')[0]+'_label.nii.gz')
        seg_nifty = nb.load(seg_path)
        seg_data=seg_nifty.get_fdata()
        for label in labels:
            organ_range=get_range(seg_data,label)
            volume=vol_data[:,:,organ_range[0]:organ_range[-1]].transpose((1,0,2))
            seg=seg_data[:,:,organ_range[0]:organ_range[-1]].transpose((1,0,2))
            seg=(seg==label).astype(np.uint8)
            vol_dict[label]=vol_dict.get(label,[])+[volume]
            seg_dict[label]=seg_dict.get(label,[])+[seg]
    return vol_dict,seg_dict

def batch_segment(model,query_slices, support_slices, support_fg_masks,support_bg_masks):
    support_slices = [[support_slice.cuda().float() for support_slice in support_slices]]
    support_fg_masks = [[support_fg_mask.cuda() for support_fg_mask in support_fg_masks]]
    support_bg_masks = [[support_bg_mask.cuda() for support_bg_mask in support_bg_masks]]
    query_slices = [query_slice.cuda().float() for query_slice in query_slices]
    query_pred= model(support_slices, support_fg_masks, support_bg_masks,query_slices)
    query_pred=query_pred.squeeze(1).argmax(dim=1)
    return query_pred.cpu().detach().numpy().transpose((1,2,0))

def retrieve(model,support_volume,support_seg,query_volume,n=10):
    pred_query_mask=np.zeros_like(query_volume)
    support_slice_num=support_volume.shape[2]
    query_slice_num=query_volume.shape[2]
    
    support_idx_list=list(range(support_slice_num))
    query_idx_list=list(range(query_slice_num))
    for i in range(n):
        query_slices=[]
        support_slices=[]
        support_fg_masks=[]
        support_bg_masks=[]
        support_idxs=support_idx_list[math.floor(i / n * support_slice_num):math.floor((i + 1) / n * support_slice_num)]
        query_idxs=query_idx_list[math.floor(i / n * query_slice_num):math.floor((i + 1) / n * query_slice_num)]
        support_idx=[support_idxs[len(support_idxs)//2]]
        for k in support_idx:
            support_slice=scale_and_normalize(support_volume[:,:,k])
            support_slice=np.repeat(np.expand_dims(support_slice,axis=2),3,axis=2)
            support_slices.append(torch.from_numpy(support_slice.transpose((2,0,1))).unsqueeze(0))
            support_fg_mask=support_seg[:,:,k].astype(np.uint8)
            support_bg_mask=1-support_fg_mask
            support_fg_masks.append(torch.from_numpy(support_fg_mask).unsqueeze(0))
            support_bg_masks.append(torch.from_numpy(support_bg_mask).unsqueeze(0))
        for j in query_idxs:
            query_slice=scale_and_normalize(query_volume[:,:,j])
            query_slice=np.repeat(np.expand_dims(query_slice,axis=2),3,axis=2)
            query_slices.append(torch.from_numpy(query_slice.transpose((2,0,1))).unsqueeze(0))
        query_mask=batch_segment(model,query_slices, support_slices, support_fg_masks,support_bg_masks)
        pred_query_mask[:,:,query_idxs[0]:query_idxs[-1]+1]=query_mask
    return pred_query_mask

def eval(model,iter,support_vol_dict,support_mask_dict,query_vol_dict,query_mask_dict,labels,eval_txt_path):
    dice_dict={}
    labels=labels
    model = model
    average_labels_dice=0
    with open(eval_txt_path,'a') as f:
        f.write("step {}: Start evaluating!\n".format(iter+1))
    for label in labels:
        for i in range(len(support_vol_dict[label])):
            for j in range(len(query_vol_dict[label])): 
                pred_query_mask=retrieve(model,support_vol_dict[label][i],support_mask_dict[label][i],query_vol_dict[label][j],n=12)
                dice=cal_dice(pred_query_mask,query_mask_dict[label][j])
                dice_dict[label]=dice_dict.get(label,[])+[dice]
        total_dice=0
        for dice in dice_dict[label]:
            total_dice+=dice
        with open(eval_txt_path,'a') as f:
            f.write("Average dice for label {} is {}\n".format(label,total_dice/len(dice_dict[label])))
            print("Average dice for label {} is {}\n".format(label,total_dice/len(dice_dict[label])))
            average_labels_dice+=total_dice/len(dice_dict[label])
    return average_labels_dice/len(labels)