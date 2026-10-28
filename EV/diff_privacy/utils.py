import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class LoadData():
    def __init__(self, df_client) -> None:
        self.df_client = df_client

    def load_data(self):
        try:
            scalerx = MinMaxScaler(feature_range=(0, 1))
            scalery = MinMaxScaler(feature_range=(0, 1))
            X = scalerx.fit_transform(self.df_client[['vehicle_speed','vehicle_acceleration','vehicle_totalEnergyConsumed']])
            Y = scalery.fit_transform(self.df_client[['vehicle_actualBatteryCapacity']])
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
            
            X_train_tensors = Variable(torch.Tensor(X_train))
            X_test_tensors = Variable(torch.Tensor(X_test))
            y_train_tensors = Variable(torch.Tensor(y_train))
            y_test_tensors = Variable(torch.Tensor(y_test)) 

            X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
            X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 
            train_ds= TensorDataset(X_train_tensors_final, y_train_tensors) 
            trainloader = DataLoader(train_ds, batch_size=len(X_train_tensors)) 
            
            test_ds= TensorDataset(X_test_tensors_final, y_test_tensors) 
            testloader = DataLoader(test_ds, batch_size=len(X_test_tensors)) 
            
            return trainloader, testloader
        
        except Exception as e: print(e)