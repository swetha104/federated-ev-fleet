import numpy as np
import timeit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import csv
import os
import torch
import torch.nn as nn
import random
from opacus import PrivacyEngine

def train(model, privacy_engine, trainloader, criterion, optimizer, cID, epochs, round_number):

        model.train()
        avg_train_loss = 0.0
        privacy_engine.attach(optimizer)
        t_start_time = timeit.default_timer()
        
        for _ in range(epochs):
            for _, (x_data, y_data) in enumerate(trainloader):
                    optimizer.zero_grad()
                    outputs = model(x_data)
                    loss = criterion(outputs, y_data)
                    loss.backward()
                    optimizer.step()
                    avg_train_loss += loss.item()
        
        avg_train_loss = avg_train_loss/len(trainloader.dataset)

        t_end_time = timeit.default_timer()
        t_elapsed = t_end_time - t_start_time
        epsilon, _ = optimizer.privacy_engine.get_privacy_spent(
            0.001
        )
        print('Training... ID : {} Dataset size : {} TrainingTime : {}s'.format(cID, len(trainloader.dataset), round(t_elapsed), 2))
        print('Training Loss ID:{} loss:{} epsilon:{}'.format(cID, avg_train_loss, epsilon))
        
        return avg_train_loss, cID, epsilon
  

              
def test(net, testloader, criterion, cID, round_number):
    
    avg_test_loss = 0.0
    net.eval()
    Xt, yt = next(iter(testloader))
    loss_function = nn.MSELoss()
    output = torch.tensor([])
    all_y = []
    with torch.no_grad():
        for X, y in testloader:
            y_star = net(X)
            output = torch.cat((output, y_star), 0)
            avg_test_loss += loss_function(y_star, y).item()
            all_y.append(y)

    avg_test_loss = avg_test_loss/len(testloader)
    yt = torch.cat(all_y, dim=0).numpy()
    ########### Write to csv file ###########
    
    data_predict = output.data.numpy()
    N = len(testloader.dataset)
    r2 = r2_score(yt, data_predict)
    adjusted_r2 = 1-((1-r2) * (N-1) / (N-3-1))
    results_dict = {'Iteration' : round_number,  
                    'Client_ID' : cID,
                    'MSE' : mean_squared_error(yt, data_predict), 
                    'RMSE' : np.sqrt(mean_squared_error(yt, data_predict)), 
                    'MAE' : mean_absolute_error(yt, data_predict),
                    'R2_Score' : r2,
                    'Adjusted_r2' :  adjusted_r2
                    }
    fieldnames = ['Iteration', 'Client_ID', 'MSE', 'RMSE', 'MAE', 'R2_Score', 'Adjusted_r2']
    #print(r2, adjusted_r2, len(testloader.dataset))
    file_exists = os.path.isfile("prediction_results_100.csv")
    with open("prediction_results_100.csv", 'a') as csv_file:  
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_dict)
        csv_file.close()

    return avg_test_loss, adjusted_r2, cID
