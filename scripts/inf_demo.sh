cur_dir=`dirname $0`
project_dir=`cd ${cur_dir} && cd .. && pwd`
#echo $project_dir
export PYTHONPATH=${project_dir}:$PYTHONPATH
#echo $PYTHONPATH
cd $project_dir
python inference_entire.py -i='./dataload/2560_test/train_vaild_split.json' -o='./heatmap/demo' -c='./ckpt/demo_ft.pth'  -m='mobilenetv2'


