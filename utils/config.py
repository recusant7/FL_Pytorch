"""
forked form https://github.com/iQua/flsim
"""

from collections import namedtuple
import json


class Config(object):
    """Configuration module."""

    def __init__(self, config):
        self.paths = ""
        # Load config file
        with open(config, 'r') as config:
            self.config = json.load(config)
        # Extract configuration
        self.extract()

    def extract(self):
        config = self.config

        # -- Clients --
        fields = ['total', 'fraction', 'label_distribution',
                  'do_test', 'test_partition']
        defaults = (0, 0, 'uniform', False, None)
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)

        assert 0 < self.clients.fraction < 1

        # -- Data --
        fields = ['loading', 'partition', 'IID', 'bias', 'shard']
        defaults = ('static', 0, True, None, None)
        params = [config['data'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.data = namedtuple('data', fields)(*params)

        # -- Federated learning --
        fields = ['rounds', 'target_accuracy', 'task', 'epochs', 'batch_size', 'lr']
        defaults = (0, None, 'train', 0, 0, 0.01)
        params = [config['federated_learning'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.fl = namedtuple('fl', fields)(*params)
        assert 0 < self.fl.target_accuracy < 1
        # -- Model --
        self.dataset = config['dataset']

        # -- Paths --
        fields = ['data', 'model', 'reports']
        defaults = ('./data', './models', None)
        params = [config['paths'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        # Set specific model path
        params[fields.index('model')] += '/' + self.dataset

        self.paths = namedtuple('paths', fields)(*params)

        # -- Server --
        self.server = config['server']


if __name__ == "__main__":
    config = Config("configs/MNIST/mnist.json")
    print(config.data.IID)
