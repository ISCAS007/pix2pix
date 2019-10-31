#which python
#conda activate new

time=`date "+%G%m%d"`
python train.py --dataroot ./datasets/facades --name sketch${time} --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode sketch --norm batch --pool_size 0 --img_root_path /home/yzbx/lf/parseData/train --seg_root_path /home/liufang/sketchCompelte2/train_sketch_GT/merge_GT