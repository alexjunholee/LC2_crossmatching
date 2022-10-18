import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as data
import pandas as pd
from os.path import join
from sklearn.neighbors import NearestNeighbors
import math
import torch
import random
import sys
import itertools
from tqdm import tqdm
import signal

def handler(signum, frame):
    tqdm.write("skipping iteration because of dataloader freezing")
    return None

default_splits = {
    # 'train': ["train"],
    'test': ["test"]
    # 'test': ["test_night"]
}


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    
class ImagesFromList(Dataset):
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = []
        isrange = []
        for imgs in self.images[idx].split(","):
            if 'disp_npy' in imgs.split("/"):  # range images should be cropped
                im = np.load(imgs).squeeze()
                im = self.transform(im[128: 3 * 128, :])
                is_r = torch.tensor(False)
            elif 'img' in imgs.split("/"):
                im = Image.open(imgs)
                im = self.transform(im)
                is_r = torch.tensor(False)
            else:
                im = np.load(imgs)
                im = self.transform(im)
                is_r = torch.tensor(True)
            img.append(im)
            isrange.append(is_r)

        if len(img) == 1:
            img = img[0]
            isrange = isrange[0]
        else:
            img = torch.stack(img, 0)
            isrange = torch.stack(isrange, 0)

        return img, isrange, idx


class VisibilityDataset(Dataset):
    def __init__(self, root_dir, splits='', nNeg=5, transform=None, mode='train',
                 posDistThr=10., negDistThr=25., cached_queries=1000, cached_negatives=1000,
                 positive_sampling=True, bs=24, threads=8, margin=0.1):

        # initializing
        assert mode in ('train', 'val', 'test')

        if splits in default_splits:
            self.splits = default_splits[splits]
        elif splits == '':
            self.splits = default_splits[mode]
        else:
            self.splits = splits.split(',')

        self.qIdx = []
        self.qImages = []
        self.pIdx = []
        self.nonNegIdx = []
        self.dbImages = []
        self.utmQ = []
        self.utmDb = []

        # hyper-parameters
        self.nNeg = nNeg
        self.margin = margin
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.cached_queries = cached_queries
        self.cached_negatives = cached_negatives

        # flags
        self.cache = None
        self.mode = mode

        # other
        self.transform = transform
        # load data
        city = self.splits[0]
        print("=====> {}".format(city))
        if mode != "test":
            qPath = join(root_dir, city)
            dbPath = join(root_dir, city)
        else:
            qPath = join(root_dir, city, 'query')
            dbPath = join(root_dir, city, 'database')

        # when GPS / UTM is available
        # load query data
        qData_d = np.loadtxt(join(qPath, "disp" + '_timelists.txt'), dtype=str)
        qData_r = np.loadtxt(join(qPath, "range" + '_timelists.txt'), dtype=str)
        # qData_r = np.loadtxt(join(qPath, "img" + '_timelists.txt'), dtype=str)
        # load database data
        dbData_d = np.loadtxt(join(dbPath, "disp" + '_timelists.txt'), dtype=str)
        # dbData_d = np.loadtxt(join(dbPath, "img" + '_timelists.txt'), dtype=str)
        dbData_r = np.loadtxt(join(dbPath, "range" + '_timelists.txt'), dtype=str)
        # arange based on task
        ext = '.npy'
        # ext = '.png'
        qSeqKeys_d = self.array_as_path(join(qPath, 'disp_npy'), qData_d, ext)
        qSeqKeys_r = self.array_as_path(join(qPath, 'range_npy'), qData_r, ext)
        # qSeqKeys_r = self.array_as_path(join(qPath, 'img'), qData_r, ext)
        # dbSeqKeys_d = self.array_as_path(join(dbPath, 'img'), dbData_d, ext)
        dbSeqKeys_d = self.array_as_path(join(dbPath, 'disp_npy'), dbData_d, ext)
        dbSeqKeys_r = self.array_as_path(join(dbPath, 'range_npy'), dbData_r, ext)

        if self.mode == 'test': # this is to adjust average distances.
            qdownsample_d = np.arange(len(qSeqKeys_d)) % 30 == 31
            qdownsample_r = np.arange(len(qSeqKeys_r)) % 10 == 0
            dbdownsample_d = np.arange(len(dbSeqKeys_d)) % 60 == 0
            dbdownsample_r = np.arange(len(dbSeqKeys_r)) % 20 == 21
            qData_d = qData_d[qdownsample_d, :]
            qData_r = qData_r[qdownsample_r, :]
            dbData_d = dbData_d[dbdownsample_d, :]
            dbData_r = dbData_r[dbdownsample_r, :]
            qSeqKeys_d = [qSeqKeys_d[i] for i in np.where(qdownsample_d)[0]]
            qSeqKeys_r = [qSeqKeys_r[i] for i in np.where(qdownsample_r)[0]]
            dbSeqKeys_d = [dbSeqKeys_d[i] for i in np.where(dbdownsample_d)[0]]
            dbSeqKeys_r = [dbSeqKeys_r[i] for i in np.where(dbdownsample_r)[0]]
        else: # as camera is 30Hz and LiDAR is 10Hz, downsample images.
            qdownsample_d = np.arange(len(qSeqKeys_d)) % 3 == 0
            qdownsample_r = np.arange(len(qSeqKeys_r)) % 1 == 0
            dbdownsample_d = np.arange(len(dbSeqKeys_d)) % 3 == 0
            dbdownsample_r = np.arange(len(dbSeqKeys_r)) % 1 == 0
            qData_d = qData_d[qdownsample_d, :]
            qData_r = qData_r[qdownsample_r, :]
            dbData_d = dbData_d[dbdownsample_d, :]
            dbData_r = dbData_r[dbdownsample_r, :]
            qSeqKeys_d = [qSeqKeys_d[i] for i in np.where(qdownsample_d)[0]]
            qSeqKeys_r = [qSeqKeys_r[i] for i in np.where(qdownsample_r)[0]]
            dbSeqKeys_d = [dbSeqKeys_d[i] for i in np.where(dbdownsample_d)[0]]
            dbSeqKeys_r = [dbSeqKeys_r[i] for i in np.where(dbdownsample_r)[0]]

        # prepare all the mixed-up variables
        self.qImages.extend(qSeqKeys_d)
        self.qImages.extend(qSeqKeys_r)
        print("Q size: {}".format(len(self.qImages)))
        self.dbImages.extend(dbSeqKeys_d)
        self.dbImages.extend(dbSeqKeys_r)
        print("db size: {}".format(len(self.dbImages)))
        qData = np.concatenate((qData_d, qData_r), axis=0)
        dbData = np.concatenate((dbData_d, dbData_r), axis=0)
        self.numDB_d = len(dbSeqKeys_d)
        self.numQ_d = len(qSeqKeys_d)
        qSeqKeys_d.extend(qSeqKeys_r)
        qSeqKeys = qSeqKeys_d

        # utm coordinates
        utmQ = np.array(qData[:, :2], dtype=float)
        utmDb = np.array(dbData[:, :2], dtype=float)
        utmDistQ = np.sum(np.linalg.norm(utmQ[:-1] - utmQ[1:], axis=1)) / (len(self.qImages)-1)
        utmDistDb = np.sum(np.linalg.norm(utmDb[:-1] - utmDb[1:], axis=1)) / (len(self.dbImages)-1)
        print("mode: {}, ".format(self.mode) + "Avg. distance Q: %.2f" % utmDistQ + ", Avg. distance Db: %.2f" % utmDistDb)
        self.utmQ = utmQ
        self.utmDb = utmDb

        # find positive images for training
        neigh = NearestNeighbors(algorithm='brute')
        neigh.fit(utmDb)
        pos_distances, pos_indices = neigh.radius_neighbors(utmQ, self.posDistThr)

        if self.mode == 'train':
            nD, nI = neigh.radius_neighbors(utmQ, self.negDistThr)

        for q_seq_idx in range(len(qSeqKeys)):
            if self.mode == 'train':
                if len(pos_indices[q_seq_idx]) > 1: # do not self-choose if there's more than one truth
                    pos_inds = np.setdiff1d(pos_indices[q_seq_idx], q_seq_idx)
                else: pos_inds = pos_indices[q_seq_idx]
                n_uniq_frame_idxs = np.unique(nI[q_seq_idx])
                self.nonNegIdx.append(n_uniq_frame_idxs)
            else:
                pos_inds = pos_indices[q_seq_idx]
            pos_ind_range = pos_inds[pos_inds >= self.numDB_d]
            pos_ind_disp = pos_inds[pos_inds < self.numDB_d]
            p_uniq_frame_idxs = np.zeros((0, ))
            if len(pos_ind_range) > 0:
                # print("range pairs: " + "%d" %len(pos_ind_range))
                p_uniq_frame_idxs = np.concatenate((p_uniq_frame_idxs, pos_ind_range))
            if len(pos_ind_disp) > 0:
                # print("disp pairs: " + "%d" %len(pos_ind_disp))
                p_uniq_frame_idxs = np.concatenate((p_uniq_frame_idxs, pos_ind_disp))
            self.qIdx.append(q_seq_idx)
            self.pIdx.append(p_uniq_frame_idxs.astype(int))

            # in training we have two thresholds, one for finding positives and one for finding images
            # that we are certain are negatives.

        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx, dtype=object)
        self.qImages = np.asarray(self.qImages)
        self.pIdx = np.asarray(self.pIdx, dtype=object)
        self.nonNegIdx = np.asarray(self.nonNegIdx, dtype=object)
        self.dbImages = np.asarray(self.dbImages)

        # decide device type ( important for triplet mining )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.threads = threads
        self.bs = bs

        if mode == 'train':

            # for now always 1-1 lookup.
            self.negCache = np.asarray([np.empty((0,), dtype=int)] * len(self.qIdx))

            # calculate weights for positive sampling
            if positive_sampling:
                self.__calcSamplingWeights__()
            else:
                self.weights = np.ones(len(self.qIdx)) / float(len(self.qIdx))

    def __calcSamplingWeights__(self):

        # length of query
        N = len(self.qIdx)

        # initialize weights
        self.weights = np.ones(N)


    def array_as_path(self, path, data, ext):
        qkeys = []
        for i in range(np.shape(data)[0]):
            qkeys.append(join(path, data[i, 2] + ext))
        return qkeys

    @staticmethod
    def filter(seqKeys, seqIdxs, center_frame_condition):
        keys, idxs = [], []
        for key, idx in zip(seqKeys, seqIdxs):
            if idx[len(idx) // 2] in center_frame_condition:
                keys.append(key)
                idxs.append(idx)
        return keys, np.asarray(idxs)

    @staticmethod
    def arange_as_seq(data, path, seq_length):

        seqInfo = pd.read_csv(join(path, 'seq_info.csv'), index_col=0)

        seq_keys, seq_idxs = [], []
        for idx in data.index:

            # edge cases.
            if idx < (seq_length // 2) or idx >= (len(seqInfo) - seq_length // 2):
                continue

            # find surrounding frames in sequence
            seq_idx = np.arange(-seq_length // 2, seq_length // 2) + 1 + idx
            seq = seqInfo.iloc[seq_idx]

            # the sequence must have the same sequence key and must have consecutive frames
            if len(np.unique(seq['sequence_key'])) == 1 and (seq['frame_number'].diff()[1:] == 1).all():
                seq_key = ','.join([join(path, 'images', key + '.jpg') for key in seq['key']])

                seq_keys.append(seq_key)
                seq_idxs.append(seq_idx)

        return seq_keys, np.asarray(seq_idxs)

    @staticmethod
    def collate_fn(batch):
        """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

        Args:
            batch: list of tuple (query, positive, negatives).
                - query: torch tensor of shape (3, h, w).
                - positive: torch tensor of shape (3, h, w).
                - negative: torch tensor of shape (n, 3, h, w).
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """

        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None, None, None, None

        query, positive, negatives, indices, q_isrange, p_isrange, n_isrange = zip(*batch)

        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
        negatives = torch.cat(negatives, 0)
        indices = list(itertools.chain(*indices))
        q_isrange = data.dataloader.default_collate(q_isrange)
        p_isrange = data.dataloader.default_collate(p_isrange)
        n_isrange = torch.cat(n_isrange, 0)

        return query, positive, negatives, negCounts, indices, q_isrange, p_isrange, n_isrange

    def __len__(self):
        return len(self.triplets)

    def new_epoch(self):

        # find how many subset we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)

        # get all indices
        arr = np.arange(len(self.qIdx))

        # apply positive sampling of indices
        arr = random.choices(arr, self.weights, k=len(arr))

        # calculate the subcache indices
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)

        # reset subset counter
        self.current_subset = 0

    def update_subcache(self, net=None, outputdim=None):

        # reset triplets
        self.triplets = []

        # if there is no network associate to the cache, then we don't do any hard negative mining.
        # Instead we just create some naive triplets based on distance.
        if net is None:
            qidxs = np.random.choice(len(self.qIdx), self.cached_queries, replace=True)

            for q in qidxs:

                # get query idx
                qidx = self.qIdx[q]

                # get positives
                pidxs = self.pIdx[q]

                # choose a random positive (within positive range (default 10 m))
                pidx = np.random.choice(pidxs, size=1)[0]

                # get negatives, make sure there's only negatives in the set.
                negIdx = np.setdiff1d(np.arange(len(self.dbImages)), self.nonNegIdx[q])
                nidxs = np.random.choice(negIdx, size=self.nNeg)

                # package the triplet and source of the pair
                triplet = [qidx, pidx, *nidxs]

                self.triplets.append(triplet)

            # increment subset counter
            self.current_subset += 1

            return

        qidxs = np.asarray(self.subcache_indices[self.current_subset])

        # take their positive in the database
        pidxs = np.unique([i for idx in self.pIdx[qidxs] for i in idx])

        # take m = 5*cached_queries is number of negative images
        nidxs = np.random.choice(len(self.dbImages), self.cached_negatives, replace=True)

        # and make sure that there is no positives among them
        nidxs = nidxs[np.in1d(nidxs, np.unique([i for idx in self.nonNegIdx[qidxs] for i in idx]), invert=True)]

        # make dataloaders for query, positive and negative images
        opt = {'batch_size': self.bs, 'shuffle': False, 'num_workers': self.threads, 'pin_memory': True}
        qloader = torch.utils.data.DataLoader(ImagesFromList(self.qImages[qidxs], transform=self.transform), **opt)
        ploader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[pidxs], transform=self.transform), **opt)
        nloader = torch.utils.data.DataLoader(ImagesFromList(self.dbImages[nidxs], transform=self.transform), **opt)

        # calculate their descriptors
        net.eval()
        with torch.no_grad():

            # initialize descriptors
            qvecs = torch.zeros(len(qidxs), outputdim).to(self.device)
            pvecs = torch.zeros(len(pidxs), outputdim).to(self.device)
            nvecs = torch.zeros(len(nidxs), outputdim).to(self.device)

            bs = opt['batch_size']

            # compute descriptors
            for i, batch in tqdm(enumerate(qloader), desc='compute query descriptors', total=len(qidxs) // bs,
                                 position=2, leave=False):
                X, isrange, _ = batch
                image_encoding = net.encoder(X.to(self.device), isrange)
                vlad_encoding = net.pool(image_encoding)
                qvecs[i * bs:(i + 1) * bs, :] = vlad_encoding
            for i, batch in tqdm(enumerate(ploader), desc='compute positive descriptors', total=len(pidxs) // bs,
                                 position=2, leave=False):
                X, isrange, _ = batch
                image_encoding = net.encoder(X.to(self.device), isrange)
                vlad_encoding = net.pool(image_encoding)
                pvecs[i * bs:(i + 1) * bs, :] = vlad_encoding
            for i, batch in tqdm(enumerate(nloader), desc='compute negative descriptors', total=len(nidxs) // bs,
                                 position=2, leave=False):
                X, isrange, _ = batch
                image_encoding = net.encoder(X.to(self.device), isrange)
                vlad_encoding = net.pool(image_encoding)
                nvecs[i * bs:(i + 1) * bs, :] = vlad_encoding

        tqdm.write('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        pScores = torch.mm(qvecs, pvecs.t())
        pScores, pRanks = torch.sort(pScores, dim=1, descending=True)

        # calculate distance between query and negatives
        nScores = torch.mm(qvecs, nvecs.t())
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)

        # convert to cpu and numpy
        pScores, pRanks = pScores.cpu().numpy(), pRanks.cpu().numpy()
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()

        tqdm.write('>> Searching for hard negative triplets...')
        # selection of hard triplets
        for q in range(len(qidxs)):

            qidx = qidxs[q]

            # find positive idx for this query (cache idx domain)
            cached_pidx = np.where(np.in1d(pidxs, self.pIdx[qidx]))

            # find idx of positive idx in rank matrix (descending cache idx domain)
            pidx = np.where(np.in1d(pRanks[q, :], cached_pidx))

            # take the closest positve
            if len(pScores[q, pidx][0]) == 0:
                dPos = 0
            else:
                dPos = pScores[q, pidx][0][0]

            # get distances to all negatives
            dNeg = nScores[q, :]

            # how much are they violating
            loss = dPos - dNeg + self.margin ** 0.5
            violatingNeg = 0 < loss

            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg:
                continue

            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]

            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]

            # select the closest positive (back to cache idx domain)
            cached_pidx = pRanks[q, pidx][0][0]

            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            pidx = pidxs[cached_pidx]
            hardestNeg = nidxs[cached_hardestNeg]

            # package the triplet and target
            triplet = [qidx, pidx, *hardestNeg]

            self.triplets.append(triplet)

        # increment subset counter
        self.current_subset += 1

    def __getitem__(self, idx):
        # get triplet
        triplet = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]

        # load images into triplet list
        if 'disp_npy' in self.qImages[qidx].split("/"): # range images should be cropped
            query = np.load(self.qImages[qidx]).squeeze()
            query = self.transform(query[128: 3 * 128, :])
            q_isrange = False
        elif 'img' in self.qImages[qidx].split("/"):
            query = Image.open(self.qImages[qidx])
            query = self.transform(query)
            q_isrange = False
        else:
            query = np.load(self.qImages[qidx])
            query = self.transform(query)
            q_isrange = True


        if 'disp_npy' in self.dbImages[pidx].split("/"):  # range images should be cropped
            positive = np.load(self.dbImages[pidx]).squeeze()
            positive = self.transform(positive[128: 3 * 128, :])
            p_isrange = False
        elif 'img' in self.dbImages[pidx].split("/"):
            positive = Image.open(self.dbImages[pidx])
            positive = self.transform(query)
            p_isrange = False
        else:
            positive = np.load(self.dbImages[pidx])
            positive = self.transform(positive)
            p_isrange = True

        negatives = []
        n_isranges = []
        for nnidx in nidx:
            if 'disp_npy' in self.dbImages[nnidx].split("/"):  # range images should be cropped
                negative = (np.load(self.dbImages[nnidx]).squeeze())
                negative = self.transform(negative[128: 3 * 128, :])
                n_isrange = torch.tensor(False)
            elif 'img' in self.dbImages[pidx].split("/"):
                negative = Image.open(self.dbImages[nnidx])
                negative = self.transform(negative)
                n_isrange = torch.tensor(False)
            else:
                negative = (np.load(self.dbImages[nnidx]))
                negative = self.transform(negative)
                n_isrange = torch.tensor(True)
            negatives.append(negative)
            n_isranges.append(n_isrange)

        negatives = torch.stack(negatives, 0)
        n_isranges = torch.stack(n_isranges, 0)
        return query, positive, negatives, [qidx, pidx] + nidx, q_isrange, p_isrange, n_isranges
