import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

class LoadData():
    def __init__(self, df_client) -> None:
        self.df_client = df_client

    def sliding_windows(self, data, seq_length):
        x = []
        y = []

        for i in range(len(data)-seq_length-1):
            _x = data[i:(i+seq_length)]
            _y = data[i+seq_length]
            x.append(_x)
            y.append(_y)
        return np.array(x),np.array(y)
    
    def load_data(self):
        sc = MinMaxScaler()
        self.df_client_data = self.df_client.iloc[:,3:].values
        training_data = sc.fit_transform(self.df_client_data)
        #print(training_data)
        seq_length = 4

        x, y = self.sliding_windows(training_data, seq_length)

        train_size = int(len(y) * 0.7)
        test_size = len(y) - train_size

        dataX = Variable(torch.Tensor(np.array(x)))
        dataY = Variable(torch.Tensor(np.array(y)))

        trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
        trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

        testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
        testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

        #print(self.df_client_data.shape)
        print(trainX.shape, trainY.shape)
        #print(dataX.shape, dataY.shape)
        #print(testX.shape, testY.shape)

        return trainX, trainY, dataX, dataY
