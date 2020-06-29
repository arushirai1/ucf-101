import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import deepcluster.clustering as clustering
import deepcluster.models as models
from deepcluster.util import AverageMeter, Logger, UnifLabelSampler
from Utils import build_paths
from Dataset import UCF10
import pickle
def parse_args():
    parser = argparse.ArgumentParser(description='Retreive Clustering assignments')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16', 'c3d'], default='alexnet',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--batch', default=30, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--verbose', action='store_true', help='chatty')

    return parser.parse_args()

def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    class_idxs, train_split, test_split, frames_root, remaining = build_paths()

    #parameters for UCF101 dataset loading
    num_classes = 101
    clip_len = 16
    dataset = UCF10(class_idxs=class_idxs, split=[train_split[0]], frames_root=frames_root,
                     clip_len=clip_len, train=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=32)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=False)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device Being Used:', device)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    model = model.to(device)

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            state_dict=checkpoint['state_dict'].copy()
            to_delete=[]
            # remove top_layer parameters from checkpoint
            for key in state_dict:
                if 'top_layer' in key:
                    to_delete.append(key)
            for key in to_delete:
                del state_dict[key]
            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))
            exit()

    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)
    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # get the features for the whole dataset
    features, labels = compute_features(dataloader, model, len(dataset))
    with open('features.pickle', 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)
    clustering_loss = deepcluster.cluster(features, verbose=args.verbose)
    train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                              dataset)
    #store clustering assignments
    with open('reassignedobj.pickle', 'wb') as f:
        pickle.dump(train_dataset, f)
    #store clustering assignments
    with open('clustering_assignments.pickle', 'wb') as f:
        pickle.dump(train_dataset.imgs, f)
    print("Train dataset shape: ", train_dataset.shape)


def compute_features(dataloader, model, N):

    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, label) in enumerate(dataloader):
        print(input_tensor.shape)
        torch.no_grad()
        input_var = torch.autograd.Variable(input_tensor.cuda())#, volatile=True)
        s_aux=time.time()
        #print(i, s_aux)
        aux = model(input_var).data.cpu().numpy()
        #print(i, time.time()-s_aux)
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
            labels = np.zeros((N))
        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
            labels[i * args.batch: (i + 1) * args.batch] = label
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux
            labels[i * args.batch:] = label

            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t' 
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features, labels

if __name__ == '__main__':
    args = parse_args()
    main(args)
