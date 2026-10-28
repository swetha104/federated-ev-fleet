import argparse
from typing import Callable, Dict, List, Optional, Tuple, OrderedDict
import flwr as fl
import numpy as np
import pandas as pd
from client import BatteryClient
from model import BatteryDataModel
from utils import LoadData
import torch
import wandb
import random

############### CLI ###############
parser = argparse.ArgumentParser()
#parser.add_argument("--data_path", type=str, required=True)
#parser.add_argument("--no_eval_clients", type=int, required=True)
parser.add_argument("--iterations", type=int, required=True) # Enter the number of iterations
args = parser.parse_args()

############### Read the csv file ###############
#df = pd.read_csv(args.data_path, delimiter=';')
#df = pd.read_csv(args.data_path, delimiter=',')
df = pd.read_csv('data/modified_data_17_09_2022.csv')
print(df.columns)
df = df[df['vehicle_id'].notna()]
df['vehicle_id'] = df['vehicle_id'].apply(int)

# Append the vehicle IDs to a list
new_client_ids = []
for i in np.unique(df['vehicle_id']):
    new_key = f'{i:.0f}'
    new_client_ids.append(new_key)

############### Pytorch model instantiation ###############
net = BatteryDataModel()

############### wandb initialization ###############
record_wandb = True
proj_name = "TEST" # Enter the project name as required

############### Select the clients for evaluation ###############
class TripCriterion_eval(fl.server.criterion.Criterion):
    '''Select the required clients for evaluation'''
    def select(self, client):
        if client.cid in evaluate_clients:
            return client.cid

############### Select the clients for training ######################
class TripCriterion_train(fl.server.criterion.Criterion):
    '''Select the required clients for training'''
    def select(self, client):
        if client.cid in train_clients:
            return client.cid

############### Custom Strategy ###############     
class CustomStrategy(fl.server.strategy.FedAvg):

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Function to configure the next round of training for the selected clients."""

        # FL round number
        config = {
            "round_number": server_round
        }

        # Initialize the wandb project here
        if record_wandb == True:
            self.run = wandb.init(project=proj_name, group="FL_SrvRnd_" + str(server_round), reinit=True)
            wandb.run.name = 'FL_SrvRnd_' + str(config["round_number"])

        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        print(config)
        fit_ins = fl.common.typing.FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        
        # Client manager
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            criterion=TripCriterion_train(), # Custom Strategy to only pick the clients necessary for training
        )

        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
        ):
        """Function to aggregate the results returned by the clients selected and requested for training in configure fit."""

        aggregated_parameters_tuple  = super().aggregate_fit(server_round, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        
        if aggregated_parameters  is not None:
            # Save weights
            print(f"Saving round {server_round} weights...")
            #print(*weights)
            #np.savez(f"output/round-{rnd}-weights.npz", *weights)
            aggregated_weights: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
  
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")
            #if record_wandb:
                #wandb.log(state_dict)
        else:
            print('None', aggregated_parameters)

         # Use the Table feature to log the data and plot the custom charts.
        if record_wandb:
            avg_train_loss = []
            evaluate_clients_fit = []
            server_l = []
            for _, r in results:
                avg_train_loss.append(r.metrics["avg_train_loss"])
                evaluate_clients_fit.append(r.metrics["cID"])
                server_l.append(server_round)
               

            data = [[x, y] for (x, y) in zip(evaluate_clients_fit, avg_train_loss)]
        
            table = wandb.Table(data=data, columns = ["Client ID", "Avg Loss"])
            wandb.log({"Avg training loss" : 
                            wandb.plot.line(table, "Client ID", "Avg Loss",
                            title="Avg training loss for the clients in configure_fit round")})


        print('Training_clients:  _____________', evaluate_clients_fit)
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
            criterion=TripCriterion_eval(), # Custom Strategy to only pick the clients necessary for evaluation
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

        if record_wandb:
            adj_r2 = []
            avg_test_loss = []
            evaluate_clients_test = []
            server_l = []
            for _, r in results:
                adj_r2.append(r.metrics["accuracy_r2"])
                avg_test_loss.append(r.metrics["avg_test_loss"])
                evaluate_clients_test.append(r.metrics["CID_test"])
                server_l.append(server_round)

            data = [[x, y] for (x, y) in zip(evaluate_clients_test, adj_r2)]

            # Use the Table feature to log the data and plot the custom charts.
            table = wandb.Table(data=data, columns = ["Client ID", "Adjusted_r2 Score"])
            wandb.log({"Adjusted_r2 Scores for Evaluation Clients" : wandb.plot.line(table, "Client ID", "Adjusted_r2 Score",
            title="Adjusted_r2 Scores for Evaluation Clients")})

            data = [[x, y] for (x, y) in zip(evaluate_clients_test, avg_test_loss)]
     
            table = wandb.Table(data=data, columns = ["Client ID", "Avg test loss"])
            wandb.log({"Avg test loss" : wandb.plot.line(table, "Client ID", "Avg test loss",
            title="Avg Test Loss for Evaluation Clients")})

            self.run.finish()

        print('Eval_clients:  _____________', evaluate_clients_test)
        # Call aggregate_evaluate from base class (FedAvg)
        params, _ = super().aggregate_evaluate(server_round, results, failures)
        return params, {"accuracy": accuracy_aggregated}
    
def client_fn(cid):
    """Function to create Client instances according to their cID"""
    #print(cid, type(cid))
    if cid:
        df_client = df[df['vehicle_id'] == int(cid)]
        ld = LoadData(df_client)
        trainloader, testloader = ld.load_data()
    else:
        pass
    return BatteryClient(net, trainloader, testloader, cid)

if __name__ == "__main__":
    
    #evaluate_clients = random.sample(list(np.unique(df['vehicle_id'])), 10)
    #evaluate_clients = [str(ev_ID) for ev_ID in evaluate_clients]
    
    ## Please note that the following clients are used only for the testing purpose. Please adapt them according to your convenience.
    train_clients_chosen_for_test = ['577', '27', '0', '758']
    evaluate_clients_chosen_for_test = ['844', '1048', '539', '540', '907', '61']

    train_clients = train_clients_chosen_for_test # Please modify them accordingly
    evaluate_clients = evaluate_clients_chosen_for_test # Please modify them accordingly 

    strategy = CustomStrategy(
        #fraction_fit=0.00469,
        fraction_fit=len(train_clients)/len(np.unique(df['vehicle_id'])),   #0.938 #0.469
        fraction_evaluate=len(evaluate_clients)/len(np.unique(df['vehicle_id'])),
        min_evaluate_clients=len(evaluate_clients),
        min_available_clients=len(new_client_ids)
    )

    ray_init_args = {"include_dashboard": True} # Initilaize Ray dashboard

    fl.simulation.start_simulation(
        client_fn=client_fn, # Client function
        clients_ids = new_client_ids,
        client_resources={"num_cpus": 0.1}, # Adjust the CPU utilization here
        config=fl.server.ServerConfig(num_rounds=args.iterations), # Number of FL rounds
        strategy=strategy,
        ray_init_args=ray_init_args,
     )