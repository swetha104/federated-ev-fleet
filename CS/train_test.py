import numpy as np
import timeit
from sklearn.metrics import mean_squared_error, r2_score
import csv
import os
import numpy as np

def train(model, trainX, trainY, criterion, optimizer, cID, epochs, round_number):
    if len(trainX) > 4:
        model.train()
        avg_train_loss = 0.0

        t_start_time = timeit.default_timer()
        
        for _ in range(epochs):
            outputs = model(trainX)
            optimizer.zero_grad()
        
            loss = criterion(outputs, trainY)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
            #if epochs % 100 == 0:
                #print("Epoch: %d, loss: %1.5f" % (epochs, loss.item()))

        avg_train_loss = avg_train_loss/len(trainX)

        t_end_time = timeit.default_timer()
        t_elapsed = t_end_time - t_start_time
        print('Training... ID : {} Dataset size : {} TrainingTime : {}s'.format(cID, len(trainX), round(t_elapsed), 2))
        print('Training Loss ID:{} loss:{}'.format(cID, avg_train_loss))

    return avg_train_loss, cID
              
def test(net, dataX, dataY, criterion, cID, round_number):
    if len(dataX) > 4:
        net.eval()
        avg_test_loss = 0.0

        output = net(dataX)
        loss = criterion(output, dataY).item()
        avg_test_loss += loss

        data_predict = output.data.numpy()
        dataY_plot = dataY.data.numpy()

        avg_test_loss = avg_test_loss/len(dataX)

        ########### Write to csv file ###########
        
        N = len(dataX)
        r2 = r2_score(dataY_plot, data_predict)
        adjusted_r2 = 1-((1-r2) * (N-1) / (N-1-1))
        results_dict = {'Iteration' : round_number,  
                        'Client_ID' : cID,
                        'MSE' : mean_squared_error(dataY_plot, data_predict), 
                        'RMSE' : np.sqrt(mean_squared_error(dataY_plot, data_predict)), 
                        'R2_Score' : r2,
                        'Adjusted_r2' :  adjusted_r2
                        }
        fieldnames = ['Iteration', 'Client_ID', 'MSE', 'RMSE', 'R2_Score', 'Adjusted_r2']
        
        file_exists = os.path.isfile("/home/swetha/08_10_2022/CS/output/prediction_results_300.csv")
        with open("/home/swetha/08_10_2022/CS/output/prediction_results_300.csv", 'a') as csv_file:  
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results_dict)
            csv_file.close()
        
        #print(avg_test_loss, adjusted_r2, cID)
        with open("/home/swetha/08_10_2022/CS/output/prediction.csv", 'a') as csv_p:  
            np.savetxt(csv_p, data_predict.flatten(), delimiter=",")
            csv_p.write("\n")
    
        with open("/home/swetha/08_10_2022/CS/output/actual.csv", 'a') as csv_a:  
            np.savetxt(csv_a, dataY_plot.flatten(), delimiter=",")
            csv_a.write("\n")

    return avg_test_loss, adjusted_r2, cID
