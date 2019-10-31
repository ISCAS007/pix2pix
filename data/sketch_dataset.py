# -*- coding: utf-8 -*-

import os
import glob
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform

class SketchDataset(BaseDataset):
    def __init__(self,opt):
        BaseDataset.__init__(self, opt)
        self.img_root_path=opt.img_root_path
        self.seg_root_path=opt.seg_root_path

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        #class name list
        self.class_names=['train_aeroplane_bird',
                          'train_bicycle_motorbike',
                          'train_bus_car',
                          'train_cat_dog_sheep',
                          'train_cow_horse']

        if opt.phase=='train':
            class_info_files=glob.glob(os.path.join('data','lists','train_*.txt'))
            #key=image filename without ext
            #value=class name {'train_bus_car','train_cat_dog_sheep'}
            self.class_info={}

            for file_name in class_info_files:
                class_name=os.path.splitext(os.path.basename(file_name))[0]
                assert class_name in self.class_names
                with open(file_name,'r') as f:
                    for l in f.readlines():
                        self.class_info[l.strip()]=class_name
        else:
            seg_files=glob.glob(os.path.join(self.seg_root_path,'*','*.png'))
            self.class_info={}

            for file_name in seg_files:
                sub_class_name=file_name.split(os.sep)[-2]
                raw_filename=os.path.splitext(file_name.split(os.sep)[-1])[0]

                raw_filename=os.path.join(sub_class_name,raw_filename)

                if sub_class_name=='airplane':
                    self.class_info[raw_filename]='train_aeroplane_bird'
                else:
                    for class_name in self.class_names:
                        if class_name.find(sub_class_name)!=-1:
                            self.class_info[raw_filename]=class_name
                            break
                    else:
                        assert False,'unknown sub class name {}'.format(sub_class_name)

        #image_full_path=os.path.join(img_root_path,raw_filenames+'-0.png')
        self.raw_filenames=[k for k in self.class_info.keys()]
        #seg_full_path=os.path.join(img_root_path,raw_filenames+'.png')

    def __len__(self):
        return len(self.raw_filenames)

    def __getitem__(self,idx):
        raw_name=self.raw_filenames[idx]

        if self.opt.phase=='train':
            img_file=os.path.join(self.img_root_path,raw_name+'-0.png')
            seg_file=os.path.join(self.seg_root_path,raw_name+'.png')
        else:
            sub_class_name,raw_name=raw_name.split(os.sep)
            img_file=os.path.join(self.img_root_path,raw_name+'-0.png')
            if not os.path.exists(img_file):
                img_file=os.path.join(self.img_root_path,sub_class_name+'_'+raw_name+'-0.png')
            seg_file=os.path.join(self.seg_root_path,sub_class_name,raw_name+'.png')
        assert os.path.exists(img_file),img_file
        assert os.path.exists(seg_file),seg_file

        AB = Image.open(img_file).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        seg=cv2.imread(seg_file,cv2.IMREAD_GRAYSCALE)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # from img_name to class_name to class_label
        class_name=self.class_info[raw_name]
        class_label=self.class_names.index(class_name)
        return {'A': A, 'B': B, 'seg_img':seg, 'A_paths': img_file, 'B_paths': img_file,
                'seg_paths':seg_file,'class_name':class_name,'class_label':class_label}

if __name__ == '__main__':
    split='test'
    img_root_path=os.path.expanduser('~/lf/parseData/{}'.format(split))
    img_files=glob.glob(os.path.join(img_root_path,'*.png'))

    #seg_root_path='/home/liufang/sketchCompelte2/train_sketches/merge_im'
    if split=='train':
        seg_root_path='/home/liufang/sketchCompelte2/train_sketch_GT/merge_GT'
        seg_files=glob.glob(os.path.join(seg_root_path,'*.png'))
    else:
        seg_root_path='/home/liufang/sketchCompelte2/test_GT'
        seg_files=glob.glob(os.path.join(seg_root_path,'*','*.png'))

    img_files.sort()
    seg_files.sort()
    N=3

    if split=='train':
        class_info_files=glob.glob(os.path.join('data','lists','train_*.txt'))
        class_info={}
        for file_name in class_info_files:
            with open(file_name,'r') as f:
                for l in f.readlines():
                    class_info[l.strip()]=os.path.splitext(os.path.basename(file_name))[0]
    else:
        class_info={}
        for file_name in seg_files:
            sub_class_name=file_name.split(os.sep)[-2]
            raw_filename=os.path.splitext(file_name.split(os.sep)[-1])[0]

            if sub_class_name in ['airplane','bird']:
                class_info[raw_filename]='train_aeroplane_bird'
            elif sub_class_name in ['bicycle','motorbike']:
                class_info[raw_filename]='train_bicycle_motorbike'
            elif sub_class_name in ['bus','car']:
                class_info[raw_filename]='train_bus_car'
            elif sub_class_name in ['cat','dog','sheep']:
                class_info[raw_filename]='train_cat_dog_sheep'
            elif sub_class_name in ['cow','horse']:
                class_info[raw_filename]='train_cow_horse'
            else:
                assert False,'unknown sub class name {}'.format(sub_class_name)


    i=0
    for k,v in class_info.items():
        i=i+1
        if i>3:
            break
        print(k,v)

    print(len(img_files),len(seg_files),len(class_info))
    for i in img_files[0:N]:
        print(i)
        key=os.path.splitext(os.path.basename(i))[0]
        if split=='test' and key.split('_')[0] in ['airplane','bird','bicycle',
                    'motorbike','bus',
                    'car','cat','dog','sheep',
                    'cow','horse']:
            sub_class_name=key.split('_')[0]
            key=key.replace(sub_class_name+'_','')

        if key[-2:]=='-0':
            key=key[:-2]
        else:
            print('bad file name',key)

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

    unknown=0
    for i,file_name in enumerate(img_files):
        key=os.path.splitext(os.path.basename(file_name))[0]

        if split=='test' and  key.split('_')[0] in ['airplane','bird','bicycle',
                    'motorbike','bus',
                    'car','cat','dog','sheep',
                    'cow','horse']:
            sub_class_name=key.split('_')[0]
            key=key.replace(sub_class_name+'_','')

        if key[-2:]=='-0':
            key=key[:-2]
        else:
            print('bad file name',key)

        if key in class_info.keys():
    #        print('class is',class_info[key])
            pass
        else:
    #        print('unknown class key',key)
            unknown+=1

        if unknown<10 and i < 10:
            print('key=',key)

    print('unknown image',unknown)

