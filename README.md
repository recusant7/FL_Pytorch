# FL_Pytorch
A simple simulation implementation of Federated learning based Pytorch. 
## The orginal paper

[Communication-Efficient Learning of Deep Networks
from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)

## Requirements
```
torch==1.5.0
torchvision==0.6.0

```
## Installation
```
git clone https://github.com/recusant7/FL_Pytorch
cd FL_Pytorch/
mkdir dataset
```
## Data partition
According to the method in the article, we randomly split the training set and partitioned into 100 clients averagely by
using this code:
```python
length = [total_sample // total_clients] * total_clients
    if IID:
        spilted_train = torch.utils.data.random_split(self.trainset, length)
```
As well, sorting the samples by label and dividing them into 200 groups are used to produce the Non-IID data. Each of 100
clients recieve 2 groups and as most clients will only have  2 kinds of digit.
## Train
### Usage
```
usage: main.py [-h] [-c CONFIG] [-l LOG] [-d DATASET]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Federated learning configuration file.
  -l LOG, --log LOG     Log messages level.
  -d DATASET, --dataset DATASET
                        the name of dataset
```
### Example
```
python main.py -c configs/MNIST/mnist.json -l info
```
### Training log
```
[INFO][15:51:44]: ----------------------round 1------------------------------
[INFO][15:51:44]: selected clients:[ 6 21 56 72 38 67 43 46 99 29]
[INFO][15:52:02]: aggregate weights
[INFO][15:52:03]: training acc: 0.9350,test acc: 0.9472, test_loss: 0.3971

[INFO][15:52:03]: ----------------------round 2------------------------------
[INFO][15:52:03]: selected clients:[66 13 55 93 37 20 67 82 47 73]
[INFO][15:52:19]: aggregate weights
[INFO][15:52:20]: training acc: 0.9692,test acc: 0.9690, test_loss: 0.0996

[INFO][15:52:20]: ----------------------round 3------------------------------
[INFO][15:52:20]: selected clients:[14 32 39 81 57 33 18 79 30 90]
[INFO][15:52:35]: aggregate weights
[INFO][15:53:13]: training acc: 0.9813,test acc: 0.9758, test_loss: 0.0785
```
## Results
Number of communication rounds to reach accuracy of 99% on test dataset of MNIST,
where B is the local minibatch size, E is the number of local epochs. The learning rate is 
sensitive, we use learning of 0.1 when the data is IID and 0.01 of non-IID.



 |      | E|  B  |IID|
 | --------   | -----:   | :----: |:----:|
 | FedAvg (paper)     | 5    |   10  |20|
 | this implementation       | 5  |  10    |14|
 | FedAvg (paper)       | 5     |   50 |29|
  | this implementation       | 5  |  50    |30|