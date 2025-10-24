#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations
import os
import argparse
import random
import json
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, average_precision_score
from matplotlib import pyplot as plt
from typing import cast, Any, Dict, List, Tuple, Optional
from collections import Counter
import csv
import copy
import gc
from utils.fcnbaseline import FCNBaseline, FCNBaselineSmall , LSTMClassifier,LSTMClassifierTeacherUnrolled,LSTMClassifier_old

from utils.data_loader import UEAloader
from utils.Inception import InceptionModel, InceptionModelNew

try:
    from sklearn.preprocessing import OneHotEncoder
    _ONEHOT_KW = {"sparse_output": False}
except TypeError:
    from sklearn.preprocessing import OneHotEncoder
    _ONEHOT_KW = {"sparse": False}


@dataclass
class InputData:
    x: torch.Tensor
    y: torch.Tensor

    def split(self, split_size: float) -> Tuple[InputData, InputData]:
        train_x, val_x, train_y, val_y = train_test_split(
            self.x.numpy(), self.y.numpy(), test_size=split_size, stratify=self.y
        )
        return (InputData(x=torch.from_numpy(train_x), y=torch.from_numpy(train_y)),
                InputData(x=torch.from_numpy(val_x), y=torch.from_numpy(val_y)))


UCR_DATASETS = ['Haptics', 'Worms', 'Computers', 'UWaveGestureLibraryAll',
                'Strawberry', 'Car', 'BeetleFly', 'wafer', 'CBF', 'Adiac',
                'Lighting2', 'ItalyPowerDemand', 'yoga', 'Trace', 'ShapesAll',
                'Beef', 'MALLAT', 'MiddlePhalanxTW', 'Meat', 'Herring',
                'MiddlePhalanxOutlineCorrect', 'FordA', 'SwedishLeaf',
                'SonyAIBORobotSurface', 'InlineSkate', 'WormsTwoClass', 'OSULeaf',
                'Ham', 'uWaveGestureLibrary_Z', 'NonInvasiveFatalECG_Thorax1',
                'ToeSegmentation1', 'ScreenType', 'SmallKitchenAppliances',
                'WordsSynonyms', 'MoteStrain', 'synthetic_control', 'Cricket_X',
                'ECGFiveDays', 'Wine', 'Cricket_Y', 'TwoLeadECG', 'Two_Patterns',
                'Phoneme', 'MiddlePhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
                'DistalPhalanxTW', 'FacesUCR', 'ECG5000', '50words', 'HandOutlines',
                'Coffee', 'Gun_Point', 'FordB', 'InsectWingbeatSound', 'MedicalImages',
                'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup',
                'SonyAIBORobotSurfaceII', 'ChlorineConcentration', 'Plane', 'Lighting7',
                'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup',
                'uWaveGestureLibrary_X', 'FaceFour', 'RefrigerationDevices', 'ECG200',
                'ToeSegmentation2', 'CinC_ECG_torso', 'BirdChicken', 'OliveOil',
                'LargeKitchenAppliances', 'uWaveGestureLibrary_Y',
                'NonInvasiveFatalECG_Thorax2', 'FISH', 'ProximalPhalanxOutlineCorrect',
                'Cricket_Z', 'FaceAll', 'StarLightCurves', 'ElectricDevices', 'Earthquakes',
                'DiatomSizeReduction', 'ProximalPhalanxTW']


###################
@staticmethod
def _to_1d_binary(y_true: np.ndarray, y_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(y_true.shape) > 1:
        return np.argmax(y_true, axis=-1), np.argmax(y_preds, axis=-1)

    else:
        return y_true, (y_preds > 0.5).astype(int)

#####################
def _fix_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Downsample to target_len if longer, else pad with zeros to target_len."""
    T, D = arr.shape
    if T == target_len:
        return arr.astype(np.float32, copy=False)
    if T > target_len:
        xt = torch.from_numpy(arr).float().T.unsqueeze(0)        # (1, D, T)
        xt_ds = F.interpolate(xt, size=target_len, mode='linear', align_corners=True)
        return xt_ds.squeeze(0).T.cpu().numpy().astype(np.float32)  # (target_len, D)
    out = np.zeros((target_len, D), dtype=np.float32)
    out[:T] = arr.astype(np.float32, copy=False)
    return out

def _dataset_to_tensors(
    ds: "UEAloader",
    encoder: Optional[OneHotEncoder] = None,
    target_len: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[np.ndarray], OneHotEncoder]:
    """
    Returns:
      x_tensor: (N, T_fixed, D) float32
      y_tensor: (N,) int64  (class ids, good for stratify)
      y_onehot: (N, C) float32 (optional; None if encoder is None and you don't need it)
      encoder: fitted or reused OneHotEncoder
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []

    for i in range(len(ds)):
        x_i, y_i = ds[i]                      # x_i: (T,D) tensor; y_i: tensor([label])
        xs.append(x_i.detach().cpu().numpy())
        ys.append(int(y_i.view(-1)[0].item()))

    # choose a fixed T
    lens = [a.shape[0] for a in xs]
    if target_len is None:
        # cap by the maximum length found (or override via opts.downsample_size if present)
        target_len = max(lens)

    # make all (T_fixed, D), then stack -> (N, T_fixed, D)
    xs_fixed = [_fix_length(a, target_len) for a in xs]
    x_tensor = torch.from_numpy(np.stack(xs_fixed, axis=0))        # (N, T_fixed, D)

    y_np = np.asarray(ys, dtype=np.int64)                          # (N,)
    y_tensor = torch.from_numpy(y_np)

    # optional one-hot (not used by InputData, but we return encoder for later use)
    y_onehot = None
    if encoder is None:
        encoder = OneHotEncoder(categories="auto", **_ONEHOT_KW)
        y_onehot = encoder.fit_transform(y_np.reshape(-1, 1))
    else:
        y_onehot = encoder.transform(y_np.reshape(-1, 1))
    y_onehot = np.asarray(y_onehot, dtype=np.float32)

    return x_tensor, y_tensor, y_onehot, encoder

# ---- main loader ----------------------------------------------------------

def load_uea_data(
    data_path: Path,
    encoder: Optional[OneHotEncoder] = None
) -> Tuple[InputData, InputData, OneHotEncoder]:
    """
    Load UEA multivariate dataset from *.ts files located in `data_path`.
    Expects: <Dataset>_TRAIN.ts and <Dataset>_TEST.ts inside that folder.

    Returns:
        train_input: InputData(x: (N_train, T_fixed, D), y: (N_train,))
        test_input : InputData(x: (N_test,  T_fixed, D), y: (N_test,))
        encoder    : fitted OneHotEncoder (if you need one-hot elsewhere)
    """
    # build datasets (UEAloader normalizes internally)
    train_ds = UEAloader(str(data_path), flag=r"_TRAIN\.ts$")
    test_ds  = UEAloader(str(data_path), flag=r"_TEST\.ts$")

    train_lens = [len(train_ds[i][0]) for i in range(len(train_ds))]
    test_lens = [len(test_ds[i][0])  for i in range(len(test_ds))]
    experiment = data_path.name
    print(
        f"dataset: {experiment} | "
        f"train_len[min/med/max]: {min(train_lens)}/{int(np.median(train_lens))}/{max(train_lens)} | "
        f"test_len[min/med/max]: {min(test_lens)}/{int(np.median(test_lens))}/{max(test_lens)}"
    )

    if (max(train_lens) > opts.downsample_size):
        print("max len is greater than downsample size, hence downsample from size", max(train_lens) ,"to a new size", opts.downsample_size)
        opts.downsample_size = opts.downsample_size
    else:
        print("max len is smaller than downsample size, hence downsmaple_size updated to max length")
        opts.downsample_size=max(train_lens)

    # materialize -> tensors with fixed length
    train_x, y_tr, y_train, encoder = _dataset_to_tensors(train_ds, encoder, target_len=opts.downsample_size)
    test_x, y_te, y_test, _       = _dataset_to_tensors(test_ds,  encoder, target_len=opts.downsample_size)

    # stats
    train_size, seq_len, n_chan = train_x.shape
    test_size = test_x.shape[0] 
    print(
        f"dataset: {experiment} | train_size: {train_size} | test_size: {test_size} | n_chan: {n_chan} | seq_len: {seq_len}"
    )
    class_distribution = dict(Counter(y_tr.tolist()))
    print("class distribution (train):", class_distribution)
    opts.output_classes= len(class_distribution)
    opts.n_channels= n_chan

    df = [opts.dataset,  train_size, test_size, seq_len,  opts.output_classes, n_chan, min(train_lens), int(np.median(train_lens)), max(train_lens),class_distribution]
    file_path = opts.common_dir + 'database_summary.csv'
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(df)

    train_input = InputData(x=train_x, y=torch.from_numpy(y_train))   
    test_input  = InputData(x=test_x, y=torch.from_numpy(y_test))

    return train_input, test_input, encoder

####################
def load_ucr_data(data_path: Path,
                  encoder: Optional[OneHotEncoder] = None
                  ) -> Tuple[InputData, InputData, OneHotEncoder]:

    experiment = data_path.parts[-1]

    train = np.loadtxt(data_path / f'{experiment}_TRAIN', delimiter=',')
    test = np.loadtxt(data_path / f'{experiment}_TEST', delimiter=',')
    
    if encoder is None:
        encoder = OneHotEncoder(categories='auto', sparse_output=False)
        y_train = encoder.fit_transform(np.expand_dims(train[:, 0], axis=-1))
    else:
        y_train = encoder.transform(np.expand_dims(train[:, 0], axis=-1))
    y_test = encoder.transform(np.expand_dims(test[:, 0], axis=-1))

    train_x = torch.from_numpy(train[:, 1:]).unsqueeze(1).float()
    test_x = torch.from_numpy(test[:, 1:]).unsqueeze(1).float()

    [train_size, n_chan, seq_len] = list(train_x.size())
    [test_size, n_chan, seq_len] = list(test_x.size())
    class_distribution = dict(Counter(train[:, 0]))
    print("database name---",opts.dataset,  "class distribution----:", class_distribution)
    opts.output_classes= len(class_distribution)
    opts.n_channels=n_chan
    df = [opts.dataset,  train_size, test_size, seq_len,  opts.output_classes, class_distribution, n_chan]
  

    file_path = opts.common_dir + 'database_summary.csv'
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(df)
        
    #downsmaple the signals
    if (seq_len>opts.downsample_size):
        train_x = F.interpolate(train_x, size=opts.downsample_size, mode='linear',align_corners=True)
        test_x = F.interpolate(test_x, size=opts.downsample_size, mode='linear',align_corners=True)
    print("downsmaple from size:", seq_len, ", to a new size:", opts.downsample_size)

    #swapp axes for LSTM : input tesnsor should be in shape : (batch, seq_len, channels)
    train_x = torch.swapaxes(train_x, 1,2)
    test_x = torch.swapaxes(test_x, 1,2)
    
    train_input = InputData(x=train_x,
                            y=torch.from_numpy(y_train))
    test_input = InputData(x=test_x,
                           y=torch.from_numpy(y_test))
    return train_input, test_input, encoder


#############################
def _load_data(experiment, data_folder, encoder) -> Tuple[InputData, InputData]:
    if 'UEA' in experiment:
        experiment_datapath = data_folder / 'Multivariate_ts' / experiment[4:]
        print("experiment_datapath",experiment_datapath)
        train, test, _ = load_uea_data(experiment_datapath)
    else:
        assert experiment in UCR_DATASETS, \
            f'{experiment} must be one of the UCR datasets: ' \
            f'https://www.cs.ucr.edu/~eamonn/time_series_data/'
        experiment_datapath = data_folder / 'UCR_TS_Archive_2015' / experiment
        if encoder is None:
            train, test, encoder = load_ucr_data(experiment_datapath)
        else:
            train, test, _ = load_ucr_data(experiment_datapath, encoder=encoder)
    return train, test
    

############################
def get_loaders(batch_size: int, mode: str, experiment: str, data_folder: str, encoder: Optional[str] = None,
                val_size: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Return dataloaders of the training / test data

    Arguments
    ----------
    batch_size:
        The batch size each iteration of the dataloader should return
    mode: {'train', 'test', 'both'}
        If 'train', this function should return (train_loader, val_loader)
        If 'test', it should return (test_loader, None)
        If 'both', it souhld return all loaders
    val_size:
        If mode == 'train', the fraction of training data to use for validation
        Ignored if mode == 'test'

    Returns
    ----------
    Tuple of (train_loader, val_loader) if mode == 'train'
    Tuple of (test_loader, None) if mode == 'test'
    Tuple of (train_loader, val_loader, test_loader) if mode == 'both'
    """
    train_data, test_data = _load_data(experiment, data_folder, encoder)
    global global_train_data
    global_train_data = train_data #save for later analysys

    assert val_size is not None, 'Val size must be defined when loading training data'
    train_data, val_data = train_data.split(val_size)

    train_loader = DataLoader(
        TensorDataset(train_data.x, train_data.y),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(
        TensorDataset(val_data.x, val_data.y),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(test_data.x, test_data.y),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )
    if mode == 'train':
        return train_loader, val_loader
    elif mode == 'test':
        return test_loader, None
    else:
        return train_loader, val_loader, test_loader


#######################
def train(net, optimizer, lr_scheduler,loader, ep):
    net.train()
    loss_all = []
    # train_iter = tqdm(loader, ncols=80)
    train_iter=loader
    for timeseries, labels in train_iter:
        timeseries, labels = timeseries.cuda(), labels.cuda()
        output = net(timeseries)
        if len(labels.shape) == 1:
                loss = F.binary_cross_entropy_with_logits(
                    output, labels.unsqueeze(-1).float(), reduction='mean'
                )
        else:
            loss = F.cross_entropy(output, labels.argmax(dim=-1), reduction='mean')
        loss_all.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # train_iter.set_description("[Train][Epoch %d] Loss: %.5f" % (ep, loss.item()))
    mean_epoch_train_loss = torch.Tensor(loss_all).mean()
    # print('[Epoch %d] Train Loss: %.5f\n' % (ep, mean_epoch_train_loss))
    return mean_epoch_train_loss

def validate(net, loader, ep):
    net.eval()
    epoch_val_loss = []
    val_iter = tqdm(loader, ncols=80)
    val_iter=loader
    for timeseries, labels in val_iter:
        timeseries, labels = timeseries.cuda(), labels.cuda()
        with torch.no_grad():
            output = net(timeseries)
            if len(labels.shape) == 1:
                val_loss = F.binary_cross_entropy_with_logits(
                    output, labels.unsqueeze(-1).float(), reduction='mean'
                ).item()
            else:
                val_loss = F.cross_entropy(output,
                                           labels.argmax(dim=-1), reduction='mean').item()
            epoch_val_loss.append(val_loss)
            
    mean_epoch_val_loss = np.mean(epoch_val_loss)
    # print('[Epoch %d] validation Loss: %.5f\n' % (ep, mean_epoch_val_loss))
    return mean_epoch_val_loss
            
def eval(net, loader, ep):
    net.eval()
    # test_iter = tqdm(loader, ncols=80)
    test_iter=loader
    outputs_all, labels_all = [], []
    test_results: Dict[str, float] = {}

    # test_iter.set_description("[Eval][Epoch %d]" % ep)
    with torch.no_grad():
        for timeseries, labels in test_iter:
            # print("sahpe of timeseries", timeseries.size())
            timeseries, labels = timeseries.cuda(), labels.cuda()
            outputs = net(timeseries)

            if len(labels.shape) == 1:
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=-1)
            outputs_all.append(outputs.cpu().detach().numpy())
            labels_all.append(labels.cpu().detach().numpy()) 
           
        true_np, preds_np = np.concatenate(labels_all), np.concatenate(outputs_all)
        test_results['roc_auc_score'] = roc_auc_score(true_np, preds_np)
        # print(f'ROC AUC score: {round(test_results["roc_auc_score"], 3)}')
        # test_results['roc_auc_score_ovr_weighted'] = roc_auc_score(true_np, preds_np,  multi_class='ovr',average='weighted')
        # test_results['roc_auc_score_ovo_weighted'] = roc_auc_score(true_np, preds_np,  multi_class='ovo',average='weighted')
        # test_results['roc_auc_score_ovr_micro'] = roc_auc_score(true_np, preds_np,  multi_class='ovr',average='micro')
        # roc_auc_score_ovr_none = roc_auc_score(true_np, preds_np,  multi_class='ovr',average=None)
        # if opts.output_classes < 2:
        #     test_results['roc_auc_score_ovr_none']  = roc_auc_score_ovr_none
        # else:
        #     test_results['roc_auc_score_ovr_none']  = ' '.join([str(x) for x in roc_auc_score_ovr_none])
            
        precision = dict()
        recall = dict()
        auc_pr = dict()
    
        if opts.output_classes < 2:
            for i in range(opts.output_classes):
                precision[i], recall[i], _ = precision_recall_curve(true_np, preds_np)
                auc_pr[i] = auc(recall[i], precision[i])
        else:
            for i in range(opts.output_classes):
                precision[i], recall[i], _ = precision_recall_curve(true_np[:, i], preds_np[:, i])
                auc_pr[i] = auc(recall[i], precision[i])
            
        average_auc_pr = sum(auc_pr.values()) / opts.output_classes
        test_results['average_auc_pr'] = average_auc_pr
        # precision["micro"], recall["micro"], _ = precision_recall_curve(true_np.ravel(), preds_np.ravel())
        # micro_precision = auc(precision["micro"], recall["micro"])
        # test_results['micro_precision'] = micro_precision
        average_precision = average_precision_score(true_np, preds_np, average="micro")
        test_results['average_precision'] = average_precision
    
        test_results['accuracy_score'] = accuracy_score(
           *_to_1d_binary(true_np, preds_np)
        )
        # print(f'Accuracy score: {round(test_results["accuracy_score"], 3)}')
        return test_results

###################################
def plot_train_history(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss,label='val_loss')
    plt.legend()
    plt.show


#######################################
def run_for_seed(args_from_main):
    global opts
    opts = args_from_main

    #eliminate randomness
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
        set_random_seed(opts.seed)


    loader_train_sample, loader_train_eval, loader_eval = get_loaders(batch_size= opts.batch, mode ='both',
                val_size=opts.val_size, experiment = opts.dataset, data_folder=opts.data_folder)

    # model = LSTMClassifier_old(opts.n_channels, opts.hidden_size, opts.num_layers, opts.output_classes).cuda()
    model=InceptionModelNew(num_blocks=6, in_channels=opts.n_channels, out_channels=32, bottleneck_channels=32, kernel_sizes=[9, 19, 39, 9, 19, 39], use_residuals=True, num_pred_classes=opts.output_classes).cuda()
    
    if opts.load is not None:
        model.load_state_dict(torch.load(opts.load))
        print("Loaded Model from %s" % opts.load)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)


    best_epoch=0
    if opts.mode == "eval":
        #calculate evaluation metrics for validation set and test set
        print("validation results:", eval(model, loader_train_eval, 0))
        test_results = eval(model, loader_eval, 0)
        print("test results:", test_results)
        return test_results
    else:
        train_loss: List[float] = []
        val_loss: List[float] = []
        best_val_loss = np.inf
        patience_counter = 0
        best_state_dict = None

        for epoch in range(1, opts.epochs+1):
            mean_epoch_train_loss = train(model, optimizer, lr_scheduler, loader_train_sample, epoch)
            mean_epoch_val_loss = validate(model, loader_train_eval, epoch)
            train_loss.append(mean_epoch_train_loss)
            val_loss.append(mean_epoch_val_loss)
        
            # print(f'Epoch: {epoch}, '
            #     f'Train loss: {train_loss[-1]}, '
            #     f'Val loss: {val_loss[-1]}')
        
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                best_state_dict =copy.deepcopy(model.state_dict())
                best_epoch= epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == opts.patience:
                    if best_state_dict is not None:
                        model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                    print('Early stopping!')
                    break
                    
        #in-case of  patience+bestepoch > total_epoch
        model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
        
        val_results = eval(model, loader_train_eval, epoch)
        print("validation results:--", val_results, "best_epoch:--", best_epoch)
        test_results = eval(model, loader_eval, epoch)
        print("final results:--", test_results)
        plt.figure()
        plot_train_history(train_loss, val_loss)

        return test_results, model
        






