import argparse
import logging
from server import Server
from utils import config

# Set up parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/MNIST/mnist.json',
                        help='Federated learning configuration file.')
    parser.add_argument('-l', '--log', type=str, default='INFO',
                        help='Log messages level.')
    parser.add_argument('-d', '--dataset', type=str, default='MNIST',
                        help='the name of dataset')

    args = parser.parse_args()
    # Set logging
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()),
        datefmt='%H:%M:%S')
    logging.info("config:{},  log:{}".format(args.config, args.log))
    # load config
    config = config.Config(args.config)
    server = Server(config)
    server.run()
