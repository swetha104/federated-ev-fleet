import argparse
from typing import List, Optional, OrderedDict
import flwr as fl
import numpy as np
import pandas as pd

from client import BatteryClient
from dataset import Dataset_all
from model import CSClass
from loaddata import LoadData

import torch
import random
import os
import csv

############### CLI ###############
parser = argparse.ArgumentParser()

parser.add_argument("--no_eval_clients", type=int, required=True)
parser.add_argument("--iterations", type=int, required=True)
args = parser.parse_args()

# Append the vehicle IDs to a list
new_client_ids = list(set(Dataset_all['chargingStation_id']))

#################### Pytorch model instantiation ####################
net = CSClass()

#################### Select the clients for evaluation ####################
class TripCriterion_eval(fl.server.criterion.Criterion):
    '''Select the required clients for evaluation'''
    def select(self, client):
        if client.cid in evaluate_clients:
            return client.cid

#################### Custom Strategy ####################     
class CustomStrategy(fl.server.strategy.FaultTolerantFedAvg):

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of training."""
        config = {
            "round_number": server_round
        }

        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        print(config)
        fit_ins = fl.common.typing.FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
        ):
        
        aggregated_parameters_tuple  = super().aggregate_fit(server_round, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        
        if aggregated_parameters  is not None:
            # Save weights
            print(f"Saving round {server_round} weights...")
            aggregated_weights: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
  
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")
        else:
            print('None', aggregated_parameters)

        avg_train_loss = []
        evaluate_clients_fit = []
        server_round_l = []
        for _, r in results:
            avg_train_loss.append(r.metrics["avg_train_loss"])
            evaluate_clients_fit.append(r.metrics["cID"])
            server_round_l.append(server_round)

        data = [[x, y, z] for (x, y, z) in zip(server_round_l, evaluate_clients_fit, avg_train_loss)]
        fieldnames = ['Iteration', 'Client_ID', 'Avg_train_loss']

        return aggregated_parameters_tuple 

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure the next round of training."""
        config = {
            "round_number": server_round
        }
        
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        eval_ins = fl.common.typing.EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            criterion=TripCriterion_eval(),
        )
      
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List,
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
       
        if not results:
            return None
        
        adjusted_r2 = [r.metrics["accuracy_r2"] * r.num_examples for _, r in results]
        number_of_samples = [r.num_examples for _, r in results]
       
        # Aggregate and print custom metric
        accuracy_aggregated = sum(adjusted_r2) / sum(number_of_samples)
        
        print(
            f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}"
        )
        
        adj_r2 = []
        avg_test_loss = []
        evaluate_clients_test = []
        server_round_l = []
        for _, r in results:
            adj_r2.append(r.metrics["accuracy_r2"])
            avg_test_loss.append(r.metrics["avg_test_loss"])
            evaluate_clients_test.append(r.metrics["CID_test"])
            server_round_l.append(server_round)

        data = [[w, x, y, z] for (w, x, y, z) in zip(server_round_l, evaluate_clients_test, avg_test_loss, adj_r2)]
        fieldnames = ['Iteration', 'Client_ID', 'Avg_train_loss', 'Adjusted_r2']
       
        # Call aggregate_evaluate from base class (FedAvg)
        params, _ = super().aggregate_evaluate(server_round, results, failures)
        return params, {"accuracy": accuracy_aggregated}
    
def client_fn(cid):
    if cid:
        df_client = Dataset_all[Dataset_all['chargingStation_id'] == cid]
        ld = LoadData(df_client)
        trainX, trainY, dataX, dataY = ld.load_data()
    else:
        pass
    return BatteryClient(net, trainX, trainY, dataX, dataY, cid)

if __name__ == "__main__":
    
    # Note : Please follow the comments for the use-case (a) here

    #evaluate_clients = random.sample(list(np.unique(Dataset_all['chargingStation_id'])), args.no_eval_clients)
    #evaluate_clients = [str(ev_ID) for ev_ID in evaluate_clients]
    evaluate_clients = ['Sind_cs_18'] # Please adapt it accordingly
    
    print("Clients chosen for evaluation", evaluate_clients)

    strategy = CustomStrategy(
        fraction_fit=1,
        fraction_evaluate=len(evaluate_clients)/len(np.unique(Dataset_all['chargingStation_id'])),
        min_evaluate_clients=len(evaluate_clients),
        min_available_clients=len(new_client_ids),
        min_completion_rate_fit = 0.1, # Enter the minimum completion rate for training
        min_completion_rate_evaluate = 0.1, # Enter the minimum completion rate for evaluation
    )

    ray_init_args = {"include_dashboard": True}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        clients_ids = new_client_ids,
        client_resources={"num_cpus": 0.1},
        config=fl.server.ServerConfig(num_rounds=args.iterations),
        strategy=strategy,
        ray_init_args=ray_init_args,
     )