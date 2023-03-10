#!/bin/bash

start_time=$(date +%s)
state_file=${start_time}.pt
curr_date=$(date +"%Y-%m-%d")

#Defaults
gpu=2
node=2
port=""
train_time="3:00"
script="train.py"
no_save=""
gtype=""
ram="8G"

while [[ "$#" -gt 0 ]]; do case $1 in
  -r|--run) run_name="$2"; shift;;
  -g|--gpu) gpu="$2"; shift;;
  -gt|--gpu-type) gtype="$2"; shift;;
  --ram) ram="$2"; shift;;
  -n|--node) node="$2"; shift;;
  -p|--port) port=$2; shift;;
  -a|--args) args="$2"; shift;;
  -t|--train) train_time="$2"; shift;;
  -ns|--no-save) no_save="true";; 
  -s|--script) script="$2"; shift;;
  -d|--date) curr_date="$2"; shift;;
  -h|--help) echo "USAGE: $0 -r <run_name> -g [ngpu|$gpu] -n [node|$node] [-a '<args>']"; exit 1;;  
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

if [ -z "$run_name" ]; then
    echo "-r,--run [name] is required."
    exit
fi

server=$(hostname)
#graham
if [[ $server == gra* ]];then
    #160 p100 2/node, 12GB, original
    #7 v100 8/node, 16GB, newer, about 50% faster than P100 and with tensor cores
    #30+6 t4   4/node, 16GB, newer, about half a V100 for compute & AI, except much slower FP64
    if [ -z "$gtype" ]; then
	    gtype="$gpu"
    else
        gtype="$gtype:$gpu"
    fi
#cedar
elif [[ $server == cdr* || $server == cedar* ]]; then
    #114 p100 4/node, 12G 
    #32  p100l 4/node, 16G 
    #192 v100l 4/node, 32G
    if [ -z "$gtype" ]; then
    	#gtype="v100l"
	gtype="$gpu"
    else
        gtype="$gtype:$gpu"
    fi
else
    echo "Unknown server: $server"
    exit
fi
echo "Using gpu: ${gtype}"

#Cedar: no more than 8 cpus per v100l
#Graham: no more than 16 cpus
output_path="${curr_date}/${run_name}"
state_file="${output_path}/${state_file}"
train_args="--output-path $output_path $args"

world_size=$(( gpu * node ))
q=$(echo $train_time | awk -F: '{ print ($1 * 60) + $2  }')
train_args="$train_args -q $q"

echo "Creating directory $output_path/logs and state file $state_file"
mkdir -p ${output_path}/logs 
echo "$port" > $state_file 

echo "Adding to queue: ${run_name}"
echo "args: ${train_args}"

#--exclusive \
sbatch --account=def-branzana \
       --mail-type=FAIL \
       --mail-user=rhythmvohra@uvic.ca \
       --exclusive \
       --nodes=$node \
       --gres=gpu:${gtype} \
       --tasks-per-node=${gpu} \
       --mem=${ram} \
       --time=0-$train_time \
       --array=1-10%1 \
       --job-name ${run_name} \
       --output=$output_path/logs/log-%N-%j.out \
       --export=ALL,tune="false",args="$train_args",world_size=$world_size,port="$port",output_path="$output_path",start_time="$start_time",state_file="$state_file",script="$script",no_save="$no_save",nodes="$node" \
       scripts/train.sh