import numpy as np
import timeit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import csv
import os
import torch
import torch.nn as nn

def train(model, trainloader, criterion, optimizer, cID, epochs, round_number):
    """Function train(....) 
            Training function - Starts the training loop
            We loop over the trainloader (data iterator) and feed the inputs to the newtork and compute the loss."""

    model.train()
    avg_train_loss = 0.0

    t_start_time = timeit.default_timer() # Start the timer 
    
    # Iterate over the data 
    for _ in range(epochs): 
        for _, (x_data, y_data) in enumerate(trainloader):
                optimizer.zero_grad()
                outputs = model(x_data)
                loss = criterion(outputs, y_data)
                loss.backward()
                optimizer.step()
                avg_train_loss += loss.item()
    
    avg_train_loss = avg_train_loss/len(trainloader.dataset) # Compute the average loss

    t_end_time = timeit.default_timer() # Stop the timer
    t_elapsed = t_end_time - t_start_time # Determine the training time
    
    print('Training... ID : {} Dataset size : {} TrainingTime : {}s'.format(cID, len(trainloader.dataset), round(t_elapsed), 2))
    print('Training Loss ID:{} loss:{}'.format(cID, avg_train_loss))
    
    return avg_train_loss, cID

              
def test(net, testloader, criterion, cID, round_number):
    """Function test(....) 
            Testing function - Starts the testing loop
            We loop over the testloader (data iterator) and feed the inputs to the newtork and compute the output."""

    avg_test_loss = 0.0
    net.eval()
    Xt, yt = next(iter(testloader))
    loss_function = nn.MSELoss()
    output = torch.tensor([])
    final_ydata = []
    with torch.no_grad():
        for X, y in testloader:
            y_star = net(X)
            output = torch.cat((output, y_star), 0)
            avg_test_loss += loss_function(y_star, y).item()
            final_ydata.append(y)

    avg_test_loss = avg_test_loss/len(testloader)
    yt = torch.cat(final_ydata, dim=0).numpy() # Output Predictions

    ########### Write to csv file ###########
    
    data_predict = output.data.numpy()
    N = len(testloader.dataset)
    r2 = r2_score(yt, data_predict)
    adjusted_r2 = 1-((1-r2) * (N-1) / (N-3-1)) # Compute the adjusted R2 scores

    results_dict = {'Iteration' : round_number,  
                    'Client_ID' : cID,
                    'MSE' : mean_squared_error(yt, data_predict), 
                    'RMSE' : np.sqrt(mean_squared_error(yt, data_predict)), 
                    'MAE' : mean_absolute_error(yt, data_predict),
                    'R2_Score' : r2,
                    'Adjusted_r2' :  adjusted_r2
                    }
    fieldnames = ['Iteration', 'Client_ID', 'MSE', 'RMSE', 'MAE', 'R2_Score', 'Adjusted_r2']
 
    # Please adapt the file path accordingly and save it in the necessary folder
    
    file_exists = os.path.isfile("prediction_results_100.csv")
    with open("prediction_results_100.csv", 'a') as csv_file:  
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_dict)
        csv_file.close()

    return avg_test_loss, adjusted_r2, cID
