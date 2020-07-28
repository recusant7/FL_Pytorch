import torch
import logging
from models import models
from utils.config import Config
from clients import Client
import copy
import numpy as np


class Server:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
        # print(self.model)
        self.clients = None
        self.client_index = []
        self.target_round = -1

    def run(self):
        self.connect_clients()
        # communication rounds
        for round in (range(1, self.config.fl.rounds + 1)):
            logging.info("-" * 22 + "round {}".format(round) + "-" * 30)
            # select clients which participate training
            selected = self.clients_selection()
            logging.info("selected clients:{}".format(selected))
            info = self.clients.train(selected)

            logging.info("aggregate weights")
            # update glob model
            glob_weights = self.fed_avg(info)
            self.model.load_state_dict(glob_weights)
            train_acc = self.getacc(info)
            test_acc, test_loss = self.test()
            logging.info(
                "training acc: {:.4f},test acc: {:.4f}, test_loss: {:.4f}\n".format(train_acc, test_acc, test_loss))
            if test_acc > self.config.fl.target_accuracy:
                self.target_round = round
                logging.info("target achieved")
                break

            # broadcast glob weights
            self.clients.update(glob_weights)

    def fed_avg(self, info):
        weights = info["weights"]
        length = info["len"]
        w_avg = copy.deepcopy(weights[0])
        for k in w_avg.keys():
            w_avg[k] *= length[0]
            for i in range(1, len(weights)):
                w_avg[k] += weights[i][k] * length[i]
            w_avg[k] = w_avg[k] / (sum(length))
        return w_avg

    def clients_selection(self):
        # randomly selection
        frac = self.config.clients.fraction
        n_clients = max(1, int(self.config.clients.total * frac))
        training_clients = np.random.choice(self.client_index, n_clients, replace=False)
        return training_clients

    def load_model(self):
        model_path = self.config.paths.model
        dataset = self.config.dataset
        logging.info('dataset: {}'.format(dataset))

        # Set up global model
        model = models.get_model(dataset)
        logging.debug(model)
        return model

    def connect_clients(self):
        self.clients = Client(self.config)
        self.client_index = self.clients.clients_to_server()
        self.clients.get_model(self.model)
        self.clients.load_data()

    def test(self):
        return self.clients.test()

    def getacc(self, info):
        corrects = sum(info["corrects"])
        total_samples = sum(info["len"])
        return corrects / total_samples


if __name__ == "__main__":
    config = Config("configs/MNIST/mnist.json")
    server = Server(config)
    server.run()
