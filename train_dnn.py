import os
from tqdm import tqdm
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from Extract_feature import extract_mfcc, extract_stft
from dnn_model import DNN1

datapath = "/data/lujd/algorithm2022/audioset/"
unique_labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
                 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

# Set Seed
seed = 111
# Python & Numpy seed
random.seed(seed)
np.random.seed(seed)
# PyTorch seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# CUDNN seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Dataloder seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# https://pytorch.org/docs/stable/notes/randomness.html


class Audio_Dataset(torch.utils.data.Dataset):
    def __init__(self, audio_df, label_dict, configs):
        super(Audio_Dataset, self).__init__()
        self.file_list = audio_df.filename.to_list()
        self.label_list = audio_df.scene_label.to_list()
        self.label_dict = label_dict

        feature_list = []
        for filename in self.file_list:
            wav_file_path = datapath + filename
            if configs["feature_type"]=="mfcc" or configs["feature_type"]=="fbanks":
                feature = extract_mfcc(
                            wav_file_path, 
                            window_duration=configs["window_duration"],
                            window_shift=configs["window_shift"],
                            n_mels=configs["n_mels"],
                            n_remain=configs["n_feature"],
                            option=configs["feature_type"]
                        )
            elif configs["feature_type"]=="stft" or configs["feature_type"]=="cutoff_stft":
                feature = extract_stft(
                            wav_file_path, 
                            window_duration=configs["window_duration"],
                            window_shift=configs["window_shift"],
                            n_remain=configs["n_feature"],
                            option=configs["feature_type"]
                        )
            if configs["model_type"]=="dnn":
                feature_list.append(feature.reshape(1,-1))             # [1, frames*n_features] (flatten)
            elif configs["model_type"]=="cnn":
                feature_list.append(np.expand_dims(feature, axis=0))   # [1, frames, n_features]
        self.feature_array = np.concatenate(feature_list, axis=0)
        print(self.feature_array.shape)

    def __getitem__(self, index):
        feature, label = self.feature_array[index], self.label_list[index]
        label = self.label_dict[label]
        return feature, label

    def __len__(self):
        return len(self.label_list)

def prepare_dataloader(configs):
    train_df = pd.read_csv(datapath+"train.csv", sep="\t")
    dev_df = pd.read_csv(datapath+"dev.csv", sep="\t")
    eval_df = pd.read_csv(datapath+"eval.csv", sep="\t")

    label_dict = {}
    for ind, c in enumerate(unique_labels):
        label_dict[c] = ind

    train_dataset = Audio_Dataset(train_df, label_dict, configs)
    dev_dataset = Audio_Dataset(dev_df, label_dict, configs)
    eval_dataset = Audio_Dataset(eval_df, label_dict, configs)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=configs["batch_size"],
            shuffle=True,
            num_workers=8,
            worker_init_fn=seed_worker
        )
    dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=configs["batch_size"],
            shuffle=False,
            num_workers=8,
            worker_init_fn=seed_worker
        )
    eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=configs["batch_size"],
            shuffle=False,
            num_workers=8,
            worker_init_fn=seed_worker
        )

    print(f"train:\t{len(train_dataset)}, {len(train_loader)}")
    print(f"dev:\t{len(dev_dataset)}, {len(dev_loader)}")
    print(f"eval:\t{len(eval_dataset)}, {len(eval_loader)}")

    return train_loader, dev_loader, eval_loader

def train_step(model, data_loader, configs, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configs["learing_rate"], weight_decay=1e-5)

    train_loss_list, true_label_list, pre_label_list = [], [], []
    # pbar = tqdm(data_loader)
    # pbar.set_description(f"Traininng")
    # for feature_list, label_list in pbar:
    for feature_list, label_list in data_loader:
        # print(f"feature size: {feature_list.shape}")
        # forward
        train_inputs = torch.Tensor(feature_list.float()).to(device)
        train_labels = torch.LongTensor(label_list).to(device)
        train_outputs = model(train_inputs)
        train_loss = criterion(train_outputs, train_labels)         # average result

        # backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        pre_labels = torch.argmax(train_outputs, dim=1)
        train_loss_list.append(train_loss.cpu().detach().numpy().item())
        true_label_list.append(train_labels.cpu().detach().numpy())
        pre_label_list.append(pre_labels.cpu().detach().numpy())

    true_label_list = np.concatenate(true_label_list, axis=0)
    pre_label_list = np.concatenate(pre_label_list, axis=0)

    train_avg_loss = sum(train_loss_list)/len(train_loss_list)
    train_accurary = np.sum(true_label_list == pre_label_list)/len(true_label_list)

    return train_avg_loss, train_accurary

def eval_step(model, data_loader, configs, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_list, true_label_list, pre_label_list = [], [], []
    # pbar = tqdm(data_loader)
    # pbar.set_description(f"Traininng")
    with torch.no_grad():
        # for feature_list, label_list in pbar:
        for feature_list, label_list in data_loader:
            # print(f"feature size: {feature_list.shape}")
            # forward
            inputs = torch.Tensor(feature_list.float()).to(device)
            true_labels = torch.LongTensor(label_list).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, true_labels)         # average result

            pre_labels = torch.argmax(outputs, dim=1)
            loss_list.append(loss.cpu().detach().numpy().item())
            true_label_list.append(true_labels.cpu().detach().numpy())
            pre_label_list.append(pre_labels.cpu().detach().numpy())

    true_label_list = np.concatenate(true_label_list, axis=0)
    pre_label_list = np.concatenate(pre_label_list, axis=0)

    avg_loss = sum(loss_list)/len(loss_list)
    accurary = np.sum(true_label_list == pre_label_list)/len(true_label_list)

    return avg_loss, accurary, true_label_list, pre_label_list

def train(model, train_loader, dev_loader, configs, device, modelpath, max_epoch=200, early_stop=3):
    
    epoch = 0
    best_epoch, best_acc = -1, -1
    train_loss_list, train_acc_list = [], []
    dev_loss_list, dev_acc_list = [], []

    while epoch < max_epoch:
        epoch += 1
        random.seed(seed + epoch)
        np.random.seed(seed + epoch)

        # train on train set
        train_loss, train_acc = train_step(model, train_loader, configs, device)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # eval on dev set
        dev_loss, dev_acc, _, _ = eval_step(model, dev_loader, configs, device)
        dev_loss_list.append(dev_loss)
        dev_acc_list.append(dev_acc)

        # save model
        if dev_acc > best_acc :
            best_epoch, best_acc = epoch, dev_acc   
            print("Epoch-{}: train loss={:.4f}, dev accurary={:.4f} | Model saving...".format(epoch, train_loss, dev_acc))
        else:
            print("Epoch-{}: train loss={:.4f}, dev accurary={:.4f} ".format(epoch, train_loss, dev_acc))

        if not os.path.exists(modelpath):
            os.makedirs(modelpath)
        new_model_name = "model_lr{}_B{}_p{}_ep{}.pkl".format(configs["learing_rate"], configs["batch_size"], configs["p_dropout"], epoch)
        torch.save(model.eval().state_dict(), modelpath + new_model_name)

        # early stop
        if epoch - best_epoch >= early_stop:
            print("EARLY STOP TRIGGERED")
            break
    
    return train_loss_list, train_acc_list, dev_loss_list, dev_acc_list

if __name__ == "__main__":
   
    configs = {
        "window_duration": 0.04,
        "window_shift": 0.02,
        "n_frames": 499,
        "n_mels": 26,
        "n_feature": 13,
        "feature_type": "fbanks",
        "learing_rate": 0.003,
        "batch_size": 256,
        "model_type": "dnn"
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, dev_loader, eval_loader = prepare_dataloader(configs["batch_size"])
    print("Data preparing done")
    model = DNN1(configs["n_frames"]*configs["n_feature"], 10).to(device)
    print("model preparing done")

    modelpath = "/data/lujd/algorithm2022/model/dnn/"
    (train_loss_list, train_acc_list,
    dev_loss_list, dev_acc_list) = train(
                                        model, train_loader, dev_loader,
                                        configs, device, modelpath, early_stop=3)
