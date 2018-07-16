# CoordConv
![](https://img.shields.io/badge/pytorch-0.4.0-blue.svg) ![](https://img.shields.io/badge/python-3.6.5-brightgreen.svg)
Pytorch implementation of CoordConv for N-D ConvLayers, and the experiments.

Reference from the paper "An intriguing failing of convolutional neural networks and the CoordConv solution."

Extends the CoordinateChannel concatenation from 2D to 1D and 3D tensors.

# Requirements
- pytorch 0.4.0
- torchvision 0.2.1
- torchsummary 1.3
- sklearn 0.19.1

# Usage
```python
from coordconv import CoordConv1d, CoordConv2d, CoordConv3d

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.coordconv = CoordConv2d(2, 32, 1, with_r=True)
        self.conv1 = nn.Conv2d(32, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64,  1, 1)
        self.conv4 = nn.Conv2d( 1,  1, 1)

    def forward(self, x):
        x = self.coordconv(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(-1, 64*64)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
```

# Experiments
Implement experiments from origin paper.

## Coordinate Classiﬁcation
Use `experiments/generate_data.py` to generate `Uniform` and `Quadrant` datasets for Coordinate Classiﬁcation task.

Use `experiments/train_and_test.py` to train and test neural network model.

### Images

|Train|Test|Predictions|
|:---:|:---:|:---:|
|![](https://i.loli.net/2018/07/16/5b4c7db11abf9.png)|![](https://i.loli.net/2018/07/16/5b4c7dbd03169.png)|![](https://i.loli.net/2018/07/16/5b4c8d88a70a2.png)|


