cur_dir=`dirname $0`
project_dir=`cd ${cur_dir} && cd .. && pwd`
#echo $project_dir
export PYTHONPATH=${project_dir}:$PYTHONPATH
#echo $PYTHONPATH
cd $project_dir
python main_entire.py --gpus 0 1 2 3 -i "./sample_neg_bags/train_vaild_split.json" -o './train_output' -l 1e-5  -sil 3 7 
