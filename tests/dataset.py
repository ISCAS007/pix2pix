# -*- coding: utf-8 -*-

import os
import glob
import matplotlib.pyplot as plt
import cv2

img_root_path=os.path.expanduser('~/lf/parseData/train')
img_files=glob.glob(os.path.join(img_root_path,'*.png'))

#seg_root_path='/home/liufang/sketchCompelte2/train_sketches/merge_im'
seg_root_path='/home/liufang/sketchCompelte2/train_sketch_GT/merge_GT'
seg_files=glob.glob(os.path.join(seg_root_path,'*.png'))

img_files.sort()
seg_files.sort()
N=3

class_info_files=glob.glob(os.path.join('data','lists','train_*.txt'))
class_info={}
for file_name in class_info_files:
    with open(file_name,'r') as f:
        for l in f.readlines():
            class_info[l]=os.path.splitext(os.path.basename(file_name))[0]

i=0
for k,v in class_info.items():
    i=i+1
    if i>3:
        break
    print(k,v)

print(len(img_files),len(seg_files))
for i in img_files[0:N]:
    print(i)
    key=os.path.splitext(os.path.basename(i))[0]
    if key in class_info.keys():
        print('class is',class_info[key])
    else:
        print('unknown class key',key)

    img=cv2.imread(i)
    plt.imshow(img)
    plt.show()

for i in seg_files[0:N]:
    print(i)
    gt=cv2.imread(i,cv2.IMREAD_GRAYSCALE)
    plt.imshow(gt)
    plt.show()

