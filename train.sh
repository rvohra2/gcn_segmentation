#!/bin/bash
#SBATCH --account=def-branzana     
#SBATCH --time=0-11:00      
#SBATCH --gres=gpu:4       #4 = no of gpus, any type 
#SBATCH --tasks-per-node=4 #= number of gpus   
#SBATCH --mem=32G        
#SBATCH --array=1-10%1 
#SBATCH --job-name=ct-bs16
#SBATCH --output=%N-%j.out    #Output from the job is redirected here

. scripts/utils.sh

module purge
module load python/3.9 scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

state_file=ct-bs16.pt
resume=""
if work_should_continue; then
    resume="--resume checkpoints/ct_r18_tt/checkpoint.pth.tar"
else
    touch $state_file
fi

# pip install --no-index -r requirements.txt 
# pip install pyclipper
#start training program
python ./instance_gcn.py $resume --state-file $state_file 

error_code=$?
echo "Program finished with error code: $error_code"

# Resubmit if not all work has been done yet.
# You must define the function work_should_continue().
if work_should_continue; then
     if [ $error_code -ne 0 ]; then
            echo "Cancelling job."
	    scancel $SLURM_ARRAY_JOB_ID
	    mv $state_file ${state_file}.error
     else
     	echo "Still have work to do on ${BASH_SOURCE[0]}:$state_file ... "
     fi 
else
     echo "WARNING: Not backing up ${output_path} to $HOME/results, will be deleted in 60 days"

     if [ ! -z "$SLURM_ARRAY_TASK_COUNT" ];then
     	scancel $SLURM_ARRAY_JOB_ID
     fi
fi
