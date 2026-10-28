from collections import OrderedDict
import flwr as fl
import torch
import warnings
import numpy as np
from opacus import PrivacyEngine

warnings.filterwarnings("ignore", category=UserWarning)
from typing import List

from train_test import train, test

# Define the Privacy parameters here
PRIVACY_PARAMS = {
    # 'target_epsilon': 5.0,
    "target_delta": 0.001,#5e-4, #50 clients
    "noise_multiplier": 1.0, #0.5, 0.75, 1.0, 1.25, 1.5.
    "max_grad_norm": 0.5,
}

# Define Flower client
class BatteryClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, idx, sample_rate) -> None:
        super().__init__()
      
        self.net = net 
        self.trainloader = trainloader
        self.testloader = testloader
        self.idx_ = idx
        self.criterion = torch.nn.MSELoss()    
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.privacy_engine = PrivacyEngine(
            self.net,
            sample_rate=sample_rate,
            target_delta=PRIVACY_PARAMS["target_delta"],
            max_grad_norm=PRIVACY_PARAMS["max_grad_norm"],
            noise_multiplier=PRIVACY_PARAMS["noise_multiplier"],
        )

    def get_parameters(self, config):
        #return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        return get_params(self.net)

    def fit(self, parameters, config):
        #self.set_parameters(parameters)
        round_number: int = config["round_number"]
        set_params(self.net, parameters)

        avg_train_loss, CID_train, eps_train = train(self.net, self.privacy_engine, self.trainloader, self.criterion, self.optimizer, 
                                        self.idx_, epochs=2000, round_number=round_number)
        
        return get_params(self.net), len(self.trainloader.dataset), {'avg_train_loss' : avg_train_loss, 'cID': CID_train, 'eps_train': eps_train}
        
    def evaluate(self, parameters, config):
        round_number: int = config["round_number"]
        try:
            #self.set_parameters(parameters)
            set_params(self.net, parameters)
            avg_test_loss, accuracy, CID_test = test(self.net, self.testloader, self.criterion, self.idx_, round_number)

        except Exception as e: print(e)
        return  avg_test_loss, len(self.testloader.dataset), {"accuracy_r2" : accuracy, "avg_test_loss" : avg_test_loss, 'CID_test' : CID_test}

def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)