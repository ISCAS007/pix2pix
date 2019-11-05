#time=`date "+%G%m%d"`
echo "name = sketch${1}"
python test.py --dataroot ./datasets/facades --name sketch${1} --model sketchGan --netG unet_256 --direction BtoA --dataset_mode sketch --norm batch --img_root_path /home/yzbx/lf/parseData/test --seg_root_path /home/liufang/sketchCompelte2/test_GT