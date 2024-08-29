PYTHON='/home/sarah/miniconda3/envs/PT/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

${PYTHON} -m methods.clustering.extract_features --dataset cifar10 --use_best_model 'True' \
 --warmup_model_dir '/home/sarah/PycharmProjects/generalized-category-discovery-main/osr_novel_categories/metric_learn_gcd/log/(28.04.2022_|_27.530)/checkpoints/model.pt'