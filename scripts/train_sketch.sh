#which python
#conda activate new

time=`date "+%G%m%d"`
python train.py --dataroot ./datasets/facades --name sketch${time} --model sketchGan --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode sketch --norm batch --pool_size 0 --batch_size 2 --load_size 280 --crop_size 256 --img_root_path /home/yzbx/lf/parseData/train --seg_root_path /home/liufang/sketchCompelte2/train_sketch_GT/merge_GT