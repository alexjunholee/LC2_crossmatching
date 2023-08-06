#!/usr/bin/env python

import argparse
import configparser
import os
from os.path import join, exists, isfile

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset.vivid import *
from lclc.models import dual_encoder
import faiss

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def compute_recall(gt, predictions, numQ, n_values, recall_str=''):
    correct_at_n = np.zeros(len(n_values))
    skips = 0
    for qIx, pred in enumerate(predictions):
        if len(gt[qIx]) == 0:
            skips += 1
            continue
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / (numQ - skips)
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall {}@{}: {:.4f}".format(recall_str, n, recall_at_n[i]))
    return all_recalls

def test(eval_set, model, device, opt, config):
    eval_set_queries = ImagesFromList(eval_set.qImages, transform=input_transform())
    eval_set_dbs = ImagesFromList(eval_set.dbImages, transform=input_transform())
    test_q_loader = DataLoader(dataset=eval_set_queries, num_workers=1,
                                  batch_size=int(config['test']['batchsize']),
                                  shuffle=False, pin_memory=(not opt.nocuda))
    test_db_loader = DataLoader(dataset=eval_set_dbs, num_workers=1,
                                  batch_size=int(config['test']['batchsize']),
                                  shuffle=False, pin_memory=(not opt.nocuda))

    model.eval()
    pool_size = model.encoder.module.enc_dim
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        q_feat = np.empty((len(eval_set_queries), pool_size), dtype=np.float32)
        db_feat = np.empty((len(eval_set_dbs), pool_size), dtype=np.float32)

        for iteration, (input_data, _, indices) in \
                enumerate(tqdm(test_q_loader, position=1, leave=False, desc='Q Iter'.rjust(15)), 1):
            indices_np = indices.detach().numpy()
            input_data = input_data.to(device)
            image_encoding = model.encoder.module.encoder_r(input_data) # input as range
            vlad_global = model.pool(image_encoding)

            q_feat[indices_np, :] = vlad_global.detach().cpu().numpy()

        for iteration, (input_data, _, indices) in \
                enumerate(tqdm(test_db_loader, position=1, leave=False, desc='DB Iter'.rjust(15)), 1):
            indices_np = indices.detach().numpy()
            input_data = input_data.to(device)
            image_encoding = model.encoder.module.encoder_d(input_data) # database as disparity
            vlad_global = model.pool(image_encoding)

            db_feat[indices_np, :] = vlad_global.detach().cpu().numpy()

    qFeat = q_feat
    pool_size = qFeat.shape[1]
    dbFeat = db_feat
    if dbFeat.dtype != np.float32:
        qFeat = qFeat.astype('float32')
        dbFeat = dbFeat.astype('float32')

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    max_n = 15
    n_values = np.arange(max_n)+1
    distance, predictions = faiss_index.search(qFeat, max_n)
    print('Calculating recalls...')
    gt = eval_set.pIdx

    correct_at_n = np.zeros(len(n_values))
    iscorrect = np.zeros(predictions.shape)
    for qIx, pred in enumerate(predictions):
        iscorrect[qIx, :] = np.in1d(pred, gt[qIx])
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / (len(eval_set.qIdx))

    print(recall_at_n)
    np.save("results/distance.npy", distance)
    np.save("results/qlocation.npy", eval_set.utmQ)
    np.save("results/Dblocation.npy", eval_set.utmDb)
    np.save("results/predictions.npy", predictions)
    np.save("results/iscorrect.npy", iscorrect)

def main():
    parser = argparse.ArgumentParser(description='LC2')
    parser.add_argument('--config_path', type=str, default='lclc/test.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--dataset_root_dir', type=str, default='/media/jhlee/4TBSSD/vivid_projects/data',
                        help='If the files in dataset_file_path are relative, use dataset_root_dir as prefix.')
    parser.add_argument('--query_seq', type=str, default='campus_day1',
                        help='The sequence to use as query')
    parser.add_argument('--db_seq', type=str, default='campus_day2',
                        help='The sequence to use as database')

    opt = parser.parse_args()
    print(opt)

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    device = torch.device("cuda")

    encoder = dual_encoder()
    encoder_dim = encoder.enc_dim

    dataset = VisibilityDataset(opt.dataset_root_dir, opt.query_seq, opt.db_seq, transform=input_transform(),
                           bs=int(config['test']['cachebatchsize']), threads=1,
                           margin=0.1, posDistThr=25)

    resume_ckpt = 'pretrained_models/dual_encoder.pth.tar'
    print(isfile(resume_ckpt))

    if not isfile(resume_ckpt):
        resume_ckpt = os.path.join(resume_ckpt)
        if not isfile(resume_ckpt):
            from download_models import download_all_models
            download_all_models(ask_for_permission=True)

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        num_clusters = str(checkpoint['state_dict']['pool.module.centroids'].shape[0])
        
        model = nn.Module()
        net_vlad = NetVLAD(num_clusters=int(num_clusters), dim=encoder_dim, vladv2=False)
        model.add_module('encoder', encoder)
        model.add_module('pool', net_vlad)

        if int(config['test']['nGPU']) > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            model.pool = nn.DataParallel(model.pool)

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

        print("=> loaded checkpoint '{}'".format(resume_ckpt, ))
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

    test(dataset, model, device, opt, config)

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs
    print('\n\nDone. Finished extracting and saving features')


if __name__ == "__main__":
    main()
