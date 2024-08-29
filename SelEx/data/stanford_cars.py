import os
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets import StanfordCars
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from config import car_root

from data.data_utils import subsample_instances

import scipy.io,csv

meta_default_path = car_root.rstrip('/') + '.mat'
csv_default_path = car_root.rstrip('/') + '.csv'

class CustomSCar(StanfordCars):

    def __init__(self, *args, **kwargs):

        super(CustomSCar, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, data_dir=car_root, transform=None, metas=meta_default_path, csvs=csv_default_path):
        metas=metas.format('annos')
        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train
        self.transform = transform

        csvreader_train = csv.reader(open(csvs.format('train')))
        csvreader_test = csv.reader(open(csvs.format('test')))

        rows_train = []
        for row in csvreader_train:
            rows_train.append(row)
            rows_train

        rows_test = []
        for row in csvreader_test:
            rows_test.append(row)
        cnt = 0
        ind_change = {}
        for i in range(1, len(rows_train)):
            cnt += 1
            ind_change[rows_train[i][1] + '$' + rows_train[i][2] + '$' + rows_train[i][3] + '$' + rows_train[i][4]] = \
                rows_train[i][-1]
        cnt = 0

        for i in range(1, len(rows_test)):
            cnt += 1
            ind_change[rows_test[i][1] + '$' + rows_test[i][2] + '$' + rows_test[i][3] + '$' + rows_test[i][4]] = \
                rows_test[i][-1]

        #if not isinstance(metas, str):
        #    raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)
        cnt = 0

        for idx, img_ in enumerate(labels_meta['annotations'][0]):

            if img_[-1] == 0:
                cnt+=1
            im_code = str(img_[1][0][0])+"$"+str(img_[2][0][0])+"$"+str(img_[3][0][0])+"$"+str(img_[4][0][0])

            if train:
                if img_[-1]==0:
                    address = data_dir.format('train')+ind_change[im_code]
                    self.data.append(address)
                    self.target.append(img_[5][0][0])
            else:
                if img_[-1]==1:
                    address = data_dir.format('test')+ind_change[im_code]
                    self.data.append(address)
                    self.target.append(img_[5][0][0])
        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)


def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_scars_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0):

    np.random.seed(seed)
    # Init entire training set
    whole_training_set = CarsDataset(data_dir=car_root, transform=train_transform, metas=meta_default_path, csvs=csv_default_path, train=True)
    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CarsDataset(data_dir=car_root, transform=test_transform, metas=meta_default_path, csvs=csv_default_path, train=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_scars_datasets(None, None, train_classes=range(98), prop_train_labels=0.5, split_train_val=False)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].target))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].target))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')