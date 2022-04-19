#Server.py

#make sure flower is installed 
import flwr as fl
fl.server.start_server(config={"num_rounds": 3})