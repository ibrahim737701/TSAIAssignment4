import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# Train data transformations
train_transforms = transforms.Compose([
   transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
   transforms.Resize((28, 28)),
   transforms.RandomRotation((-15., 15.), fill=0),
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,)),
   ])

# Test data transformations
test_transforms = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
   ])


train_data = datasets.MNIST('../data', train=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False,  transform=test_transforms)

batch_size = 512
kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)


batch_data, batch_label = next(iter(train_loader))
fig = plt.figure()
for i in range(12):
 plt.subplot(3,4,i+1)
 plt.tight_layout()
 plt.imshow(batch_data[i].squeeze(0), cmap='gray')
 plt.title(batch_label[i].item())
 plt.xticks([])
 plt.yticks([])



class Net(nn.Module):
   #This defines the structure of the NN.
   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
       self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
       self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
       self.fc1 = nn.Linear(4096, 50, bias=False)
       self.fc2 = nn.Linear(50, 10, bias=False)
   def forward(self, x):
       x = F.relu(self.conv1(x), 2)
       x = F.relu(F.max_pool2d(self.conv2(x), 2))
       x = F.relu(self.conv3(x), 2)
       x = F.relu(F.max_pool2d(self.conv4(x), 2))
       x = x.view(-1, 4096)
       x = F.relu(self.fc1(x))
       x = self.fc2(x)
       return F.log_softmax(x, dim=1)
   
