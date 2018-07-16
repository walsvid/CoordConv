import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F

from torch.autograd import Variable
# from torchsummary import summary

datatype = 'uniform'
assert datatype in ['uniform', 'quadrant']

if datatype == 'uniform':
    # Load the one hot datasets
    train_onehot = np.load('data-uniform/train_onehot.npy').astype('float32')
    test_onehot = np.load('data-uniform/test_onehot.npy').astype('float32')

    # (N, C, H, W) <=== 数据格式
    # make the train and test datasets
    # train
    pos_train = np.where(train_onehot == 1.0)
    X_train = pos_train[2]
    Y_train = pos_train[3]
    train_set = np.zeros((len(X_train), 2, 1, 1), dtype='float32')
    for i, (x, y) in enumerate(zip(X_train, Y_train)):
        train_set[i, 0, 0, 0] = x
        train_set[i, 1, 0, 0] = y

    # test
    pos_test = np.where(test_onehot == 1.0)
    X_test = pos_test[2]
    Y_test = pos_test[3]
    test_set = np.zeros((len(X_test), 2, 1, 1), dtype='float32')
    for i, (x, y) in enumerate(zip(X_test, Y_test)):
        test_set[i, 0, 0, 0] = x
        test_set[i, 1, 0, 0] = y

    train_set = np.tile(train_set, [1, 1, 64, 64])
    test_set = np.tile(test_set, [1, 1, 64, 64])

    # Normalize the datasets
    train_set /= (64. - 1.)  # 64x64 grid, 0-based index
    test_set /= (64. - 1.)  # 64x64 grid, 0-based index

    print('Train set : ', train_set.shape, train_set.max(), train_set.min())
    print('Test set : ', test_set.shape, test_set.max(), test_set.min())

    # Visualize the datasets

    plt.imshow(np.sum(train_onehot, axis=0)[0, :, :], cmap='gray')
    plt.title('Train One-hot dataset')
    plt.show()
    plt.imshow(np.sum(test_onehot, axis=0)[0, :, :], cmap='gray')
    plt.title('Test One-hot dataset')
    plt.show()

else:
    # Load the one hot datasets and the train / test set
    train_set = np.load('data-quadrant/train_set.npy').astype('float32')
    test_set = np.load('data-quadrant/test_set.npy').astype('float32')

    train_onehot = np.load('data-quadrant/train_onehot.npy').astype('float32')
    test_onehot = np.load('data-quadrant/test_onehot.npy').astype('float32')

    train_set = np.tile(train_set, [1, 1, 64, 64])
    test_set = np.tile(test_set, [1, 1, 64, 64])

    # Normalize datasets
    train_set /= train_set.max()
    test_set /= test_set.max()

    print('Train set : ', train_set.shape, train_set.max(), train_set.min())
    print('Test set : ', test_set.shape, test_set.max(), test_set.min())

    # Visualize the datasets

    plt.imshow(np.sum(train_onehot, axis=0)[0, :, :], cmap='gray')
    plt.title('Train One-hot dataset')
    plt.show()
    plt.imshow(np.sum(test_onehot, axis=0)[0, :, :], cmap='gray')
    plt.title('Test One-hot dataset')
    plt.show()

# flatten the datasets
train_onehot = train_onehot.reshape((-1, 64 * 64)).astype('int64')
test_onehot = test_onehot.reshape((-1, 64 * 64)).astype('int64')

# model definition

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

# summary(net, input_size=(2, 64, 64))
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#          AddCoords-1            [-1, 5, 64, 64]               0
#             Conv2d-2           [-1, 32, 64, 64]             192
#        CoordConv2d-3           [-1, 32, 64, 64]              96
#             Conv2d-4           [-1, 64, 64, 64]           2,112
#             Conv2d-5           [-1, 64, 64, 64]           4,160
#             Conv2d-6            [-1, 1, 64, 64]              65
#             Conv2d-7            [-1, 1, 64, 64]               2
# ================================================================
# Total params: 6,627
# Trainable params: 6,627
# Non-trainable params: 0
# ----------------------------------------------------------------

train_tensor_x = torch.stack([torch.Tensor(i) for i in train_set])
train_tensor_y = torch.stack([torch.LongTensor(i) for i in train_onehot])

train_dataset = utils.TensorDataset(train_tensor_x,train_tensor_y)
train_dataloader = utils.DataLoader(train_dataset, batch_size=32, shuffle=False)

test_tensor_x = torch.stack([torch.Tensor(i) for i in test_set])
test_tensor_y = torch.stack([torch.LongTensor(i) for i in test_onehot])

test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y)
test_dataloader = utils.DataLoader(test_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

criterion = cross_entropy_one_hot

epochs = 10

def train(epoch, net, train_dataloader, optimizer, criterion, device):
    net.train()
    iters = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        iters += len(data)
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, iters, len(train_dataloader.dataset),
                100. * (batch_idx + 1) / len(train_dataloader), loss.data.item()), end='\r', flush=True)
    print("")


for epoch in range(1, epochs + 1):
    train(epoch, net, train_dataloader, optimizer, criterion, device)


def test(net, test_loader, optimizer, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        output = net(data)
        test_loss += criterion(output, target).item()
        _, pred = output.max(1, keepdim=True)
        _, label = target.max(dim=1)
        correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test(net, test_dataloader, optimizer, criterion, device)
