cur_dir=`dirname $0`
project_dir=`cd ${cur_dir} && cd .. && pwd`
#echo $project_dir
export PYTHONPATH=${project_dir}:$PYTHONPATH
#echo $PYTHONPATH
cd $project_dir
python main_entire.py --gpus=3, -i="./data/bags_512/train_vaild_split.json" -o='./train_output' -l 1e-5  
