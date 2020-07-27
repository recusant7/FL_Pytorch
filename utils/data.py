from torchvision import datasets, transforms
from torch.utils.data import random_split,DataLoader

class MNIST:
    def __init__(self, config):
        self.config = config
        print(self.config.paths.data)
        self.path = str(self.config.paths.data) + '/' + self.config.dataset
        self.trainset=None
        self.testset=None

    def load_data(self, IID=True):
        self.trainset = datasets.MNIST(
            self.path, train=True, download=True, transform=transforms.Compose([
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.testset = datasets.MNIST(
            self.path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        if IID:
            total_clients=self.config.clients.total
            total_sample=self.trainset.data.shape[0]
            length=[total_sample//total_clients]*total_clients
            spilted_train=random_split(self.trainset,length)
            return spilted_train,self.testset
        else:
            pass

def get_data(dataset,config):
    if dataset=="MNIST":
        return MNIST(config).load_data()