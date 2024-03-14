import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import Compose
import transforms as T
import mmap

class FWIDataset(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file 
    instead of the batch used during training.

    Args:
        anno: path to annotation file
        preload: whether to load the whole dataset into memory
        sample_ratio: downsample ratio for seismic data
        file_size: # of samples in each npy file
        transform_data|label: transformation applied to data or label
    '''
    def __init__(self, anno, preload=True, sample_ratio=1, file_size=1000,
                    transform_data=None, transform_label=None, trunc = True):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        self.trunc = trunc
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        # if preload: 
        #     self.data_list, self.label_list = [], []
        #     for batch in self.batches: 
        #         data, label = self.load_every(batch)
        #         # print(data.shape, label.shape)
        #         self.data_list.append(data)
        #         self.label_list.append(label)

    # def load_every(self, batch):
    #     batch = batch.split('\t')
    #     data_path, label_path = batch[0], batch[1][:-1]
    #     print("batch", batch[1])
    #     print("data_path", data_path)
    #     print("label_path", label_path)
    #     if self.trunc:
    #         data = np.load(data_path, mmap_mode="r")[:, :, 140::self.sample_ratio, :]    
    #     else:
    #             data = np.load(data_path, mmap_mode="r")[:, :, ::self.sample_ratio, :]
    #     label = np.load(label_path)
    #     data = data.astype('float32')
    #     label = label.astype('float32')
    #     return data, label

    def __getitem__(self, idx):
        batch = self.batches[idx].split('\t')
        data_path, label_path = batch[0], batch[1][:-1]

        print("batch", batch[1])
        print("data_path", data_path)
        print("label_path", label_path)

        # data = np.load(data_path, mmap_mode='r')
        # print(data.shape)

        if self.trunc:
            data = np.load(data_path, mmap_mode='r')[:, :, 140::self.sample_ratio, :]
        else:
            data = np.load(data_path, mmap_mode='r')[:, :, ::self.sample_ratio, :]
        
        # try:
        #     with open(data_path, "r+b") as f:
        #         # Use mmap to open the file with memory mapping
        #         mmapped_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        #         data = np.load(mmapped_data, mmap_mode='r')[0, 0, 635::self.sample_ratio, 2]
        # except mmap.error as e:
        #     # Handle the error, e.g., print an error message or take appropriate action
        #     print(f"Memory mapping error: {e}")
        #     data = None  # You can set data to None or handle it differently

        label = np.load(label_path)

        data = data.astype('float32')
        label = label.astype('float32')

        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label:
            label = self.transform_label(label)

        return data, label
        
    # def __getitem__(self, idx):
    #     batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
    #     #print(batch_idx, sample_idx)
    #     if self.preload:
    #         data = self.data_list[batch_idx][sample_idx]
    #         label = self.label_list[batch_idx][sample_idx]
    #     else:
    #         data, label = self.load_every(self.batches[batch_idx])
    #         data = data[sample_idx]
    #         label = label[sample_idx]
    #     if self.transform_data:
    #         data = self.transform_data(data)
    #     if self.transform_label:
    #         label = self.transform_label(label)
    #     return data, label
        
    def __len__(self):
        return len(self.batches) * self.file_size

class FWIDataset_withTumor(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file 
    instead of the batch used during training.

    Args:
        anno: path to annotation file
        preload: whether to load the whole dataset into memory
        sample_ratio: downsample ratio for seismic data
        file_size: # of samples in each npy file
        transform_data|label: transformation applied to data or label
    '''
    def __init__(self, anno, preload=True, sample_ratio=1, file_size=1000,
                    transform_data=None, transform_label=None):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        if preload: 
            self.data_list, self.label_list, self.tumor_list = [], [], []
            for batch in self.batches: 
                data, label, tumor = self.load_every(batch)
                self.data_list.append(data)
                self.label_list.append(label)
                self.tumor_list.append(tumor)

    def load_every(self, batch):
        batch = batch.split('\t')
        data_path, label_path, tumor_path = batch[0], batch[1], batch[2][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        label = np.load(label_path)
        tumor = np.load(tumor_path)
        data = data.astype('float32')
        label = label.astype('float32')
        tumor = tumor.astype('float32')
        return data, label, tumor
        
    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx]
            tumor = self.tumor_list[batch_idx][sample_idx]
        else:
            data, label, tumor = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx]
            tumor = tumor[sample_idx]
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label:
            label = self.transform_label(label)
        return data, label, tumor
        
    def __len__(self):
        return len(self.batches) * self.file_size


if __name__ == '__main__':
    transform_data = Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(2000, 6000)
    ])
    suffix = '3_357_raw'
    dataset = FWIDataset(f'relevant_files/flat_transform_{suffix}.txt',
                transform_data=transform_data, transform_label=transform_label)
    train_set, valid_set = random_split(dataset, [2000, 1000], generator=torch.Generator().manual_seed(0))
    print('Before saving: ', len(train_set), len(valid_set))
    save = True
    if save:
        print('Saving...')
        torch.save(train_set, f'relevant_files/flat_transform_train_{suffix}.pth')
        torch.save(valid_set, f'relevant_files/flat_transform_valid_{suffix}.pth')
        print('Verifying...')
        train_set_verify = torch.load(f'relevant_files/flat_transform_train_{suffix}.pth')
        valid_set_verify = torch.load(f'relevant_files/flat_transform_valid_{suffix}.pth')
        print('Load saving: ', type(train_set_verify), len(train_set_verify), 
                                type(valid_set_verify), len(valid_set_verify))
        data, label = train_set_verify[0]
        print('Read sample: ', data.shape, label.shape)
