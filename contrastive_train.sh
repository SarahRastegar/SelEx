PYTHON='/home/sarah/miniconda3/envs/PT/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=/home/sarah/PycharmProjects/generalized-category-discovery-main/osr_novel_categories/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

# Run from methods/contrastive_training (config.py paths are relative to it). Per-dataset settings come from
# dataset_configs in config.py; every other hyperparameter is the paper default.
cd "$(dirname "$0")/methods/contrastive_training"
${PYTHON} contrastive_training.py --dataset_name 'cifar10' \
> ${SAVE_DIR}logfile_${EXP_NUM}.out
