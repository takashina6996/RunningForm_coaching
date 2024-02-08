import flwr as fl

#def weightes_average(metrics):
  #  accuracies = (num_examples * m[])

fl.server.start_server(
    server_address = "0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5000),
    strategy=fl.server.strategy.FedAvg(),
    )
