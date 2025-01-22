'''
Build trainining/testing datasets
'''
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'ETHZ_Food-101':
        if is_train == "train":
            root = os.path.join(args.data_path, 'train')
            dataset = datasets.ImageFolder(root, transform=transform)
        elif is_train == "val":
            root = os.path.join(args.data_path, 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        elif is_train == "test":
            root = os.path.join(args.data_path, 'test')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 101
    elif args.data_set == 'UEC_Food-256':
        if is_train=="train":
            root = os.path.join(args.data_path, 'train')
            dataset = datasets.ImageFolder(root, transform=transform)
        elif is_train=="val":
            root = os.path.join(args.data_path, 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        elif is_train=="test":
            root = os.path.join(args.data_path, 'test')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 256
    elif args.data_set == 'Vireo_Food-172':
        if is_train == "train":
            root = os.path.join(args.data_path, 'train')
            dataset = datasets.ImageFolder(root, transform=transform)
        elif is_train == "val":
            root = os.path.join(args.data_path, 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        elif is_train == "test":
            root = os.path.join(args.data_path, 'test')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 172
    elif args.data_set == 'SuShi-50':
        if is_train == "train":
            root = os.path.join(args.data_path, 'train')
            dataset = datasets.ImageFolder(root, transform=transform)
        elif is_train == "val":
            root = os.path.join(args.data_path, 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        elif is_train == "test":
            root = os.path.join(args.data_path, 'test')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 50

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train=="train":
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if args.finetune:
        t.append(
            transforms.Resize((args.input_size, args.input_size),
                                interpolation=3)
        )
    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=3),
            )
            t.append(transforms.CenterCrop(args.input_size))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
