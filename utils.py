import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models_imagenet
from models.vit import ViT
from torch.utils.data import Subset
from timm.models.vision_transformer import VisionTransformer, _cfg
from functools import partial
from randomaug import RandAugment
from generate_val_idx import *
import clip
import math

import numpy as np
import random
import os
import time
import models
import sys

def set_seed(seed=1): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

################################ datasets #######################################

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

################################ training & evaluation #######################################

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image


def eval_model(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

def get_datasets(args):
    if args.datasets == 'CIFAR10':
        print ('CIFAR10 dataset')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize,
            ])

        transform_test = transforms.Compose([
                transforms.ToTensor(), normalize,
            ])
            
        if args.randaug:
            print ('use randaug')
            N=2; M=14;
            transform_train.transforms.insert(0, RandAugment(N, M))
            
        if args.finetune:
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(), normalize, 
                ])
            transform_test = transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(), normalize,
                ])
            
        train_dataset = datasets.CIFAR10(root='./datasets/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root='./datasets/', train=False, transform=transform_test)
        
    elif args.datasets == 'CIFAR100':
        print ('CIFAR100 dataset')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize,
            ])

        transform_test = transforms.Compose([
                transforms.ToTensor(), normalize,
            ])
            
        if args.randaug:
            print ('use randaug')
            N=2; M=14;
            transform_train.transforms.insert(0, RandAugment(N, M))
            
        if args.finetune:
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(), normalize, 
                ])
            transform_test = transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(), normalize,
                ])
            
        train_dataset = datasets.CIFAR100(root='./datasets/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR100(root='./datasets/', train=False, transform=transform_test)

    elif args.datasets == 'ImageNet':
        print ('ImageNet dataset')
        traindir = os.path.join('/opt/data/common/ILSVRC2012/', 'train')
        valdir = os.path.join('/opt/data/common/ILSVRC2012/', 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        if args.arch == 'clip-vit-b/32':
            base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)
            train_dataset = datasets.ImageFolder(
                traindir,
                preprocess
                )
            test_dataset = datasets.ImageFolder(
                valdir,
                preprocess 
                )
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        
    print (args.randomseed)
    np.random.seed(args.randomseed)
    N = len(train_dataset)
    random_permutation = list(np.random.permutation(N))
    nn = int(N * args.val_ratio / 100.)
    print ('val_ratio:', args.val_ratio, 'nn:', nn)
    
    if args.arch == 'clip-vit-b/32' or args.datasets == 'ImageNet' and args.val_ratio > 0:
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)
        idx_val = np.where(idxs)[0]
        idxs = (1 - idxs).astype('int')
        idx_train = np.where(idxs)[0]
    elif nn == 0:
        idx_train = random_permutation
        idx_val = None
    else:
        idx_train = random_permutation[:-nn]
        idx_val = random_permutation[-nn:]
        
    # print (len(idx_train), len(idx_val))
    # print (max(idx_train))
    
    train_set = Subset(train_dataset, idx_train)
    val_set = Subset(train_dataset, idx_val)
    test_set = test_dataset
    
    if args.ddp:
        batch_size_per_GPU = args.batch_size // args.world_size
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        if nn == 0:
            val_sampler = None
        else:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size_per_GPU, sampler=train_sampler,
            num_workers=args.workers, pin_memory=True)
        
        if nn == 0:
            val_loader = None
        else:
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size_per_GPU, sampler=val_sampler,
                num_workers=args.workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size_per_GPU, sampler=test_sampler,
            num_workers=args.workers)
    else:
        print ('not ddp')
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        if nn == 0:
            val_loader = None
        else:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def bn_update(loader, model):
    model.train()
    for i, (input, target) in enumerate(loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output = model(input_var)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits

def get_model(args):
    print('Model: {}'.format(args.arch))
    
    size = 32
    if args.datasets == 'CIFAR10':
        num_classes = 10
    if args.datasets == 'CIFAR10-small':
        num_classes = 10
    elif args.datasets == 'CIFAR100':
        num_classes = 100
    elif args.datasets == 'tinyImageNet':
        num_classes = 200
        size = 64
    elif args.datasets == 'Flowers102':
        num_classes = 102
    elif args.datasets == 'ImageNet':
        num_classes = 1000
        
    if args.arch == 'clip-vit-b/32':
        base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)
        model = ModelWrapper(base_model, 512, num_classes, normalize=True)
        for p in model.parameters():
            p.data = p.data.float()
        return model

    if args.arch == 'vit-s/32':
        # import timm
        # model = timm.create_model('vit_small_patch32_224')
        # return model
        import models.vit_img as vit_img
        return vit_img.SimpleViT(1000)
        
    if args.arch == 'ViT':
        if args.datasets == 'tinyImageNet':
            model = ViT(image_size = 64, patch_size = 8, num_classes = num_classes,
                dim = int(512), depth = 6, heads = 8,
                mlp_dim = 512, dropout = 0.1, emb_dropout = 0.1)
        else:
            model = ViT(image_size = size, patch_size = 4, num_classes = num_classes,
                dim = int(512), depth = 6, heads = 8,
                mlp_dim = 512, dropout = 0.1, emb_dropout = 0.1)
        return model
    
    if args.arch == 'resnet18':
        from models import resnet
        return resnet.resnet18()
    
    if args.datasets == 'ImageNet':
        import torchvision.models as models
        return models.__dict__[args.arch]()
    
    if 'deit' in args.arch:
        # model = VisionTransformer(
        #     patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, drop_rate=args.drop,
        #     drop_path_rate=args.drop_path
        # )
        
        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, drop_rate=args.drop,
            drop_path_rate=args.drop_path
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        
        if args.datasets != 'ImageNet':
            checkpoint["model"].pop('head.weight')
            checkpoint["model"].pop('head.bias')

        model.load_state_dict(checkpoint["model"],strict=False)
        return model
    
    model_cfg = getattr(models, args.arch)

    return model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        

