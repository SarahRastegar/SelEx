# SelEx: Self-Expertise in Fine-Grained Generalized Category Discovery
<!--

<p align="center">
    <a href="https://openreview.net/forum?id=m0vfXMrLwF&noteId=m0vfXMrLwF"><img src="https://img.shields.io/badge/-ECCV%2024-blue"></a>
    <a href="https://arxiv.org/abs/2310.19776"><img src="https://img.shields.io/badge/arXiv-2310.19776-red"></a>
</p>
-->
<p align="center">
	SelEx: Self-Expertise in Fine-Grained Generalized Category Discovery (ECCV 2024)<br>
  By
  <a href="https://sarahrastegar.github.io/">Sarah Rastegar</a>, 
  <a href="https://smsd75.github.io/">Mohammadreza Salehi</a>, 
  <a href="https://yukimasano.github.io/">Yuki Asano</a>, 
  <a href="https://hazeldoughty.github.io/">Hazel Doughty</a>, and 
  <a href="https://www.ceessnoek.info/">Cees Snoek</a>.
</p>

<!-- ![image](assets/selex.png)




## Dependencies

```
pip install -r requirements.txt
```

## Config

Set paths to datasets, pre-trained models and desired log directories in ```config.py```

Set ```SAVE_DIR``` (logfile destination) and ```PYTHON``` (path to python interpreter) in ```bash_scripts``` scripts.

## Datasets

We use fine-grained benchmarks in this paper, including:                                                                                                                    
                                                                                                                                                                  
* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet](https://image-net.org/download.php)


## Scripts

**Train representation**:

```
bash bash_scripts/contrastive_train.sh
```

**Extract features**: Extract features to prepare for semi-supervised k-means. 
It will require changing the path for the model with which to extract features in ```warmup_model_dir```

```
bash bash_scripts/extract_features.sh
```

**Fit semi-supervised k-means**:

```
bash bash_scripts/k_means.sh
```


## <a name="cite"/> :clipboard: Citation

If you use this code in your research, please consider citing our paper:

```
@inproceedings{RastegarECCV2024,
title = {SelEx: Self-Expertise in Fine-Grained Generalized Category Discovery},
author = {Sarah Rastegar and Mohammadreza Salehi and Yuki M Asano and Hazel Doughty and Cees G M Snoek},
year = {2024},
booktitle = {European Conference on Computer Vision},
}
```

## Acknowledgements

The codebase is mainly built on the repo of https://github.com/sgvaze/generalized-category-discovery.-->
