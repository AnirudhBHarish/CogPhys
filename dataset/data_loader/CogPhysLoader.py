"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC-rPPG, PURE, SCAMPS, BP4D+, and UBFC-PHYS.

"""
import glob
import os
import re
from math import ceil
from scipy import signal
from scipy import sparse
from unsupervised_methods import utils
import math
import multiprocessing as mp

# To be used only for preparing data - for detecting face with YOLO5Face
try:
    mp.set_start_method('spawn', force=True)
    # print("spawned")
except RuntimeError:
    pass

import cv2
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
# from retinaface import RetinaFace   # Source code: https://github.com/serengil/retinaface

from dataset.data_loader.BaseLoader import BaseLoader

class CogPhysLoader(BaseLoader):
    def __init__(self, name, data_path, config_data, device=None):
        """Inits dataloader with lists of files.

        Args:
            name(str): name of the dataloader.
            data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.inputs = {}
        self.labels = {}
        self.name = name
        self.data_path = data_path
        self.cached_path = config_data.DATA_PATH
        # self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        # self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data
        self.device = device

        # Load data path based on the folds
        self.fold_num = int(config_data.FOLD.FOLD_NAME.split('_')[-1]) # TODO: Fix this
        self.fold_path = config_data.FOLD.FOLD_PATH

        # Load preprocessed data
        self.input_keys = config_data.COGPHYS.INPUT
        self.label_keys = config_data.COGPHYS.LABEL
        self.input_preproc = config_data.COGPHYS.INPUT_TYPE
        self.label_preproc = config_data.COGPHYS.LABEL_TYPE
        self.seq_len = config_data.COGPHYS.SEQ_LENGTH
        self.ret_dict = config_data.COGPHYS.RET_DICT
        self.example_key = self.input_keys[0]

        # samp_freq
        self.target_fs = config_data.FS
        self.input_fs = config_data.COGPHYS.INPUT_FS
        self.label_fs = config_data.COGPHYS.LABEL_FS

        self.load_preprocessed_data()

        print('Cached Data Path', self.cached_path, end='\n\n')
        print('Data Path', self.data_path, end='\n\n')
        # print('File List Path', self.file_list_path)
        print(f"{self.name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs[self.example_key])
    
    @torch.no_grad()
    def preproc_get_item_data(self, data):
        for key, preproc_list in zip(self.input_keys, self.input_preproc):
            for preproc in preproc_list:
                if preproc == 'NormAndFloat':
                    data[key] = self.norm_and_float_data(data[key], key)
                elif preproc == 'DiffNormalized':
                    data[key] = self.diff_normalize_data(data[key])
                elif preproc == 'Standardize':
                    data[key] = self.standardized_data(data[key])
                elif preproc == 'Raw':
                    pass
                elif preproc == 'AddChannel':
                    data[key] = data[key].unsqueeze(0)
                elif preproc == 'Downsample':
                    data[key] = data[key][::(self.input_fs[self.input_keys.index(key)]  // self.target_fs)]
                else:
                    raise ValueError(f'Unsupported Preprocessing Type! Got *{preproc}*')
        return data

    @torch.no_grad()
    def preproc_get_item_label(self, data):
        for key, preproc_list in zip(self.label_keys, self.label_preproc):
            for preproc in preproc_list:
                if preproc == 'NormAndFloat':
                    raise NotImplementedError('NormAndFloat not implemented for labels')
                    # data[key] = self.norm_and_float_label(data[key])
                elif preproc == 'DiffNormalized':
                    data[key] = self.diff_normalize_label(data[key])
                elif preproc == 'Standardize':
                    data[key] = self.standardized_label(data[key])
                elif preproc == 'Raw':
                    pass
                elif preproc == 'Downsample':
                    data[key] = data[key][::(self.label_fs[self.label_keys.index(key)]  // self.target_fs)]
                else:
                    raise ValueError(f'Unsupported Preprocessing Type! Got *{preproc}*')
        return data

    @torch.no_grad()
    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        # from the self.inputs dict and self.lavels dict, get the data and label for the index
        example_file = self.inputs[self.example_key][index]
        # get the folder in which the file is located
        participant_task = example_file.split(os.sep)[-2]
        # get the chunk_id of the file
        chunk_id = example_file.split(os.sep)[-1][:-4].split('_')[-1]
        # Prepare the data and label paths
        data_path = {key: self.inputs[key][index] for key in self.input_keys}
        label_path = {key: self.labels[key][index] for key in self.label_keys}
        data = {}
        label = {}
        # Use torch since operating on GPUs much faster for preprocessing
        for inp_key in self.input_keys:
            data[inp_key] = torch.Tensor(np.load(data_path[inp_key]).astype(int)).to(self.device)
        for lab_key in self.label_keys:
            label[lab_key] = torch.Tensor(np.load(label_path[lab_key]).astype(float)).to(self.device)
        # Preprocess the data and label
        data = self.preproc_get_item_data(data)
        label = self.preproc_get_item_label(label)
        # Return the data and label
        for key in self.input_keys:
            # Add channel if data is 3D
            if data[key].ndim == 3:
                data[key] = data[key].unsqueeze(-1)
            if data[key].ndim == 4 and data[key].shape[-1] == 1:
                data[key] = torch.cat([data[key], data[key], data[key]], dim=-1)
            # Permute the data to the correct format
            if self.data_format == 'NDCHW':
                data[key] = torch.permute(data[key], (0, 3, 1, 2))
                concat_dim = 1
            elif self.data_format == 'NCDHW':
                data[key] = torch.permute(data[key], (3, 0, 1, 2))
                concat_dim = 0
            elif self.data_format == 'NDHWC':
                pass
                concat_dim = -1
            else:
                raise ValueError('Unsupported Data Format!')
        if not self.ret_dict:
            data = torch.cat([data[key].cpu() for key in self.input_keys], dim=concat_dim)
            label = torch.cat([label[key].cpu() for key in self.label_keys], dim=-1)
        return data, label, participant_task, chunk_id

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        all_inputs = os.listdir(self.data_path)
        # Load files from the folds in pickle file
        with open(self.fold_path, 'rb') as f:
            fold_dict = pickle.load(f)
            chosen_fold = fold_dict[self.fold_num]
        inputs_ids = chosen_fold[self.name] # train/test/valid based on name
        input_folders = [f for f in all_inputs if f.split('_')[0] in inputs_ids]
        # Read files from the folders in input_folders: Structure "folder/key_*.npy"
        for key in self.input_keys:
            self.inputs[key] = []
            for folder in input_folders:
                all_files = sorted(glob.glob(os.path.join(self.data_path, folder, f"{key}_*.npy")))
                self.inputs[key].extend(all_files)
        for key in self.label_keys:
            self.labels[key] = []
            for folder in input_folders:
                all_files = sorted(glob.glob(os.path.join(self.data_path, folder, f"{key}_*.npy")))
                self.labels[key].extend(all_files)
        # Exclude the follow if NIR (blank frames)
        exclude_nir = ["v19_still"]
        if "nir" in self.input_keys:
            print(f"Excluding {exclude_nir} files from the dataset due to corrupted nir video")
            for key in self.input_keys:
                self.inputs[key] = [i for i in self.inputs[key] if not any(ex in i for ex in exclude_nir)]
            for key in self.label_keys:
                self.labels[key] = [i for i in self.labels[key] if not any(ex in i for ex in exclude_nir)]
        # Exclude if respiration (corrupted signal)
        exclude_resp = ["v9_still", "v7_still", "v5_still", "v31_still", "v30_still", 
                            "v15_still", "v12_still", "v11_still", "v10_still"]
        if "respiration" in self.label_keys:
            print(f"Excluding {exclude_resp} files from the dataset due to corrupted respiration signal")
            for key in self.input_keys:
                self.inputs[key] = [i for i in self.inputs[key] if not any(ex in i for ex in exclude_resp)]
            for key in self.label_keys:
                self.labels[key] = [i for i in self.labels[key] if not any(ex in i for ex in exclude_resp)]
        # if 'thermal_below" in self.input_key, only keep the files with "still" or "rest" in the name
        if self.name == "train" and ("thermal_below" in self.input_keys or "thermal_above" in self.input_keys):
            print(f"Keeping only still and rest samples")
            for key in self.input_keys:
                self.inputs[key] = [i for i in self.inputs[key] if any(ex in i for ex in ["still", "rest"])]
            for key in self.label_keys:
                self.labels[key] = [i for i in self.labels[key] if any(ex in i for ex in ["still", "rest"])]
        # make sure that the input and label files are the same other than the key
        for key in self.input_keys:
            assert self.inputs[key] == [i.replace(self.example_key, key) 
                                            for i in self.inputs[self.example_key]]
        for key in self.label_keys:
            assert self.labels[key] == [i.replace(self.example_key, key) 
                                            for i in self.inputs[self.example_key]]
        self.preprocessed_data_len = len(self.inputs[self.example_key])

    @torch.no_grad()
    def norm_and_float_data(self, data, key):
        """Converts data to float."""
        # Everything is in torch.Tensor. Normlize accordingly if uint 8 or 16. 
        if key == 'rgb_left' or key == 'rgb_right':
            data = data.float() / 255.0
        elif key == 'nir' or key == 'thermal_above' or key == 'thermal_below':
            data = data.float() / 65535.0
        else:
            data = data.float()
        return data

    @torch.no_grad()
    def diff_normalize_data(self, data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_data = torch.zeros((n, h, w, c), dtype=torch.float32)
        diffnormalized_data[:-1] = (data[1:] - data[:-1]) / (data[1:] + data[:-1] + 1e-7)
        diffnormalized_data[:-1] = diffnormalized_data[:-1] / torch.std(diffnormalized_data[:-1])
        diffnormalized_data[torch.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    @torch.no_grad()
    def diff_normalize_label(self, label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diffnormalized_label = torch.zeros(label.shape, dtype=torch.float32)
        diff_label = torch.diff(label, axis=0)
        diffnormalized_label[:-1] = diff_label / torch.std(diff_label)
        diffnormalized_label[torch.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @torch.no_grad()
    def standardized_data(self, data):
        """Z-score standardization for video data."""
        data = data - torch.mean(data)
        data = data / torch.std(data)
        data[torch.isnan(data)] = 0
        return data

    @torch.no_grad()
    def standardized_label(self, label):
        """Z-score standardization for label signal."""
        label = label - torch.mean(label)
        label = label / torch.std(label)
        label[torch.isnan(label)] = 0
        return label

    @torch.no_grad()
    def resample_ppg(self, input_signal, target_length):
        """Samples a PPG sequence into specific length."""
        return np.interp(
            torch.linspace(
                1, input_signal.shape[0], target_length), torch.linspace(
                1, input_signal.shape[0], input_signal.shape[0]), input_signal)

