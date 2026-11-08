# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '../../data/datasets/cifar10'
cifar_100_root = '../../data/datasets/cifar100'
cub_root = '../../data/datasets/CUB'
car_root = '../../data/datasets/stanford_car/'  # cars_train/, cars_test/, devkit/ (SimGCD layout)
pets_root = '../../data/datasets/pets/'

aircraft_root = '../../data/datasets/aircraft/fgvc-aircraft-2013b'
herbarium_dataroot = '../../data/datasets/herbarium_19/'
imagenet_root = '../../data/datasets/ImageNet/'#ILSVRC12'

# OSR Split dir
osr_split_dir = '../../data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path2 = '../../pretrained_models/dino/dinov2_vitb14_reg4_pretrain.pth'#

dino_pretrain_path = '../../pretrained_models/dino/dino_vitbase16_pretrain.pth'
dino_v1 = True     # backbone for training and evaluation: True = DINO ViT-B/16, False = DINOv2 ViT-B/14 reg4
warmup_pretrain_path = '../../pretrained_models/GCDTeacher/model_best.pth'
feature_extract_dir = 'osr_novel_categories/extracted_features_public_impl'     # Extract features to this directory
exp_root = 'osr_novel_categories/'          # All logs and checkpoints will be saved here

# -----------------
# PER-DATASET SETTINGS
# -----------------
# Settings of the fixed code (GitHub issue #11), the same for DINOv1 and DINOv2; everything else uses the argparse
# defaults (batch size 128, lr 0.1, weight decay 1e-4, sup_con_weight 0.35). A value given on the command line overrides this table.
dataset_configs = {
    'cub':          dict(unsupervised_smoothing=0.75, temperature=0.25, grad_from_block=10, epochs=50, unbalanced=False),
    'scars':        dict(unsupervised_smoothing=0.75, temperature=0.5,  grad_from_block=9,  epochs=50, unbalanced=False),
    'aircraft':     dict(unsupervised_smoothing=0.75, temperature=0.5,  grad_from_block=9,  epochs=50, unbalanced=False),
    'pets':         dict(unsupervised_smoothing=0.75, temperature=0.25, grad_from_block=10, epochs=50, unbalanced=False),
    'herbarium_19': dict(unsupervised_smoothing=0.75, temperature=0.5,  grad_from_block=9,  epochs=50, unbalanced=True),
    'cifar10':      dict(unsupervised_smoothing=0.5,  temperature=0.15, grad_from_block=10, epochs=20, unbalanced=False),
    'cifar100':     dict(unsupervised_smoothing=0.5,  temperature=0.15, grad_from_block=10, epochs=20, unbalanced=False),
    'imagenet_100': dict(unsupervised_smoothing=0.5,  temperature=0.15, grad_from_block=10, epochs=20, unbalanced=False),
}
