import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd
import sys,os

from benchmark import Benchmark

NUM_CLASS = 13
#NUM_SAMPLES= 4642
NUM_SAMPLES= 9993
NUM_EPOCH =50

benchmark = Benchmark()
benchmark_list = benchmark.get_benchmark_list()

train_input_file = sys.argv[1]
test_input_file   = sys.argv[2]

class IrisDataset(Dataset):
    def __init__(self, path, delimiter=','):
        self.__data = np.genfromtxt(path, delimiter=delimiter).astype(np.float32)

    def __getitem__(self, index):
        instance = self.__data[index,:]
        data = torch.from_numpy(instance[:-1])
        label = torch.from_numpy(np.array(instance[-1]).astype(int))
#        freq= torch.from_numpy(np.array(instance[1]).astype(int))

        return data, label

    def __len__(self):
        return self.__data.shape[0]

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        hidden= NUM_SAMPLES+1000
        self.fc1 = nn.Linear(NUM_SAMPLES,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,NUM_CLASS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

if __name__ == '__main__':
    train_dataloader = DataLoader(dataset=IrisDataset(train_input_file),batch_size=10, shuffle=False)
    test_dataloader = DataLoader(dataset=IrisDataset(test_input_file),batch_size=10, shuffle=True)

    epochs = NUM_EPOCH
    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        print("epoch = ",epoch)
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            instances, labels = data
            optimizer.zero_grad()

            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        #print("Len dataloader =",len(dataloader))
        #print("running loss = ",running_loss)
            if i % 100 == 99:
                # every 1000 mini-batches...
                # ...log the running loss
                print("Training loss {} Steps: {}".format(running_loss / 100, epoch * len(train_dataloader) + i))
                running_loss = 0.0

    # Save model
    filename = 'model.pt'
    counter = 0
    while os.path.exists(filename):
        filename = 'model_' + str(counter) + '.pt'
        counter += 1
    torch.save(model.state_dict(), filename)

    # For confustion matrix
    y_pred = []
    y_true = []

    for instances, labels in test_dataloader:
        output = model(instances)
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*NUM_CLASS, index = [i for i in benchmark_list],
                             columns = [i for i in benchmark_list])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
    #as_matrix = accuracy_score(y_true, y_pred)

    #instances, labels = next(iter(dataloader))
    #print("instances[0] =",instances[0])
    #print("labels[0] =",labels[0])
    #instance = instances[0].view(1,4642 )
    #label = labels[0].view(1, 1)
    #model.eval()
    #print(torch.exp(model(instance)), label)
