import flwr as fl
from collections import OrderedDict
from centralized import load_model, train, test
import torch
import search_optimal_form
import data_loading

def set_parameters(model, parameters):
   params_dict = zip(model.state_dict().keys(), parameters)
   state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
   model.load_state_dict(state_dict, strict=True)
   return model

net = load_model()
#trainloader, testloader = load_data()
trainloader, testloader, standard_train, standard_test = data_loading.load(all_data = 'personal_data/all_members_data_original.csv', test_data = '3km_testdata.csv')

class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

  def fit(self, parameters, config):
    set_parameters(net, parameters)
    train(net, trainloader, epochs=1)
    return self.get_parameters(config={}), len(trainloader.dataset), {}

  def evaluate(self, parameters, config):
    set_parameters(net, parameters)
    loss, accuracy = test(net, testloader)
    print(type(loss))
    return float(loss), len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
  server_address="127.0.0.1:8080",
  client=FlowerClient()
  )




