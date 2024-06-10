#! /usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
import pandas as pd
import sys
import os
import torch

from benchmark import Benchmark
import getopt

from env import myEnv

myEnv = myEnv()
# custom directory definition
root_dir = myEnv.root_dir
result_dir = myEnv.result_dir


# train_input_file = sys.argv[1]
# test_input_file = sys.argv[2]
def handling_options():
    options, remainder = getopt.getopt(
        sys.argv[1:],
        "r",
        [
            "retrain",
        ],
    )
    retrain = False

    for opt, arg in options:
        if opt in ("-r", "--retain"):
            retrain = True
    return retrain


def model_training(model, train_dataloader, num_epochs, filename):
    print("model Training....")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            instances, labels = data
            optimizer.zero_grad()

            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                # every 1000 mini-batches...
                # ...log the running loss
                print(
                    "Training loss {} Steps: {}".format(
                        running_loss / 100, epoch * len(train_dataloader) + i
                    )
                )
                running_loss = 0.0
    # Save model
    print(f"model stored to {filename}")
    torch.save(model.state_dict(), filename)

    return model


class MyDataset(Dataset):
    def __init__(self, df):
        data = df.to_numpy().astype(np.float32)
        data[:, 1:] = preprocessing.normalize(data[:, 1:], norm="l2")
        self.__data = data

    def __getitem__(self, index):
        instance = self.__data[index, :]
        data = torch.from_numpy(instance[1:])
        label = torch.from_numpy(np.array(instance[0]).astype(int))
        return data, label

    def __len__(self):
        return self.__data.shape[0]


class Classifier(nn.Module):
    def __init__(self, num_samples, num_class):
        super().__init__()
        hidden = num_samples + 1000
        self.fc1 = nn.Linear(num_samples, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


def drawing_confusion_matrix(uni_uniq_kernel, num_class, y_pred, y_true, test, target):
    cmap = "mako"
    # cmap = "Blues"
    cf_matrix = confusion_matrix(y_true, y_pred)
    ratio_matrix = cf_matrix.astype("float") / cf_matrix.sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(
        ratio_matrix,
        index=[i for i in te_uniq_kernel],
        columns=[i for i in te_uniq_kernel],
    )
    # remove=["SqueezeNet_Executefire4expand3x3","SqueezeNet_Executefire5expand3x3","LSTM_ExecuteLSTM"]
    # for i in remove:
    #     df_cm = df_cm.drop(i, axis=0)
    #     df_cm = df_cm.drop(i, axis=1)

    fig, ax = fig, ax = plt.subplots(figsize=(num_class, num_class))

    hm = sn.heatmap(
        data=df_cm,
        cmap=cmap,
        annot=True,
        annot_kws={"fontsize": 10},
        ax=ax,
        vmin=0,
        vmax=1,
        square=True,
        # cbar=False,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.7},
    )

    fig.tight_layout()

    # Output_dir
    # fig.savefig(output_file+".png", format='png',bbox_inches='tight',dpi=600)
    t = time.localtime()
    current_time = time.strftime("%d%m%Y_%H%M%S", t)
    output_file = result_dir / f"confusion_matrix_{test}_{target}_{current_time}"
    fig.savefig(str(output_file) + ".pdf", format="pdf", bbox_inches="tight", dpi=600)
    print(f"Confusion matrix is saved to {output_file}.pdf")


if __name__ == "__main__":
    retrain = handling_options()

    # create multiple train_file and test_file pairs
    dev_names = [
        "Intel_UHD770",
        "Nvidia_GTX_1660_Ti",
        "Nvidia_RTX_3060",
        "Nvidia_RTX_2080",
    ]

    for train_tgt in dev_names:
        for test_tgt in dev_names:
            print(f"Train: {train_tgt}, Test: {test_tgt}")
            train_file = f"{train_tgt}_wide_result.csv"
            test_file = f"{test_tgt}_wide_result.csv"

            # test_file= train_file
            train_input_file = result_dir / train_file
            test_input_file = result_dir / test_file

            train_df = pd.read_csv(train_input_file)
            train_df = train_df.fillna(0)
            train_df = train_df.drop(columns="Kernel")
            train_shape = train_df.shape

            test_df = pd.read_csv(test_input_file)
            test_df = test_df.fillna(0)
            test_df = test_df.drop(columns="Kernel")
            test_shape = test_df.shape

            # padd 0 to make same shape of train and test
            if train_shape[1] > test_shape[1]:
                test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
            elif train_shape[1] < test_shape[1]:
                train_df = train_df.reindex(columns=test_df.columns, fill_value=0)
            assert (
                test_df.shape[1] == train_df.shape[1]
            ), "test and train data should have same number of samples"

            label = "Benchmark"
            # Get common Kernel name
            tr_uniq_kernel = train_df[label].unique()
            te_uniq_kernel = test_df[label].unique()
            uni_uniq_kernel = list(set(tr_uniq_kernel).union(te_uniq_kernel))

            # mapping_dict = {kernel: idx for idx, kernel in enumerate(uni_uniq_kernel)}
            # reverse_mapping_dict = {v: k for k, v in mapping_dict.items()}
            # print(mapping_dict)
            # print(reverse_mapping_dict)
            # test_df[label] = test_df[label].replace(mapping_dict)
            label_encoder = LabelEncoder()
            train_df[label] = label_encoder.fit_transform(train_df[label])
            test_df[label] = label_encoder.fit_transform(test_df[label])

            train_dataloader = DataLoader(
                dataset=MyDataset(train_df), batch_size=50, shuffle=False
            )
            test_dataloader = DataLoader(
                dataset=MyDataset(test_df), batch_size=50, shuffle=True
            )

            num_class = len(uni_uniq_kernel)
            num_sample = test_df.shape[1] - 1  # need to remove kernel column
            num_epochs = 1000
            filename = f"model_{train_tgt}_{test_tgt}.pt"
            model = Classifier(num_sample, num_class)
            model = torch.nn.DataParallel(model)
            if retrain or not os.path.isfile(filename):
                model = model_training(model, train_dataloader, num_epochs, filename)
            else:
                model.load_state_dict(torch.load(filename))

            # For confustion matrix
            y_pred = []
            y_true = []

            for instances, labels in test_dataloader:
                output = model(instances)
                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(output)  # Save Prediction

                labels = labels.data.cpu().numpy()
                y_true.extend(labels)  # Save Truth

            report_dict = classification_report(
                y_true, y_pred, target_names=uni_uniq_kernel, output_dict=True
            )
            report_df = pd.DataFrame(report_dict)
            report_df.to_csv(result_dir / f"report_{train_tgt}_{test_tgt}.csv")

            drawing_confusion_matrix(
                uni_uniq_kernel, num_class, y_pred, y_true, test_tgt, train_tgt
            )
