import numpy as np
import torch
from time import time
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

print("libraries imported")

class MyDataset(Dataset):
    """
    This class is made so that we can use DataLoader function on csv files.
    It inherits Dataset class from torch.utils.data.
    """

    def __init__(self, isTrain):
        """
        This function reads csv files using numpy function and assigns variable
        and target data to respective attributes of the object.
        :param isTrain: if the bool value is true , it loads training data set
        or else loads testing data set.
        """

        if isTrain:
            xyI = np.loadtxt('newDataset/training-images.csv', delimiter=',', dtype=np.float32, skiprows=0)
            xyL = np.loadtxt('newDataset/training-labels.csv', delimiter=',', dtype=np.int, skiprows=0)
        else:
            xyI = np.loadtxt('newDataset/testing-images.csv', delimiter=',', dtype=np.float32, skiprows=0)
            xyL = np.loadtxt('newDataset/testing-labels.csv', delimiter=',', dtype=np.int, skiprows=0)

        self.n_samples = xyI.shape[0]
        self.x_data = torch.from_numpy(xyI)
        self.y_data = torch.from_numpy(xyL)
        self.y_data = self.y_data.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


trainset = MyDataset(True)
valset = MyDataset(False)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=True)

print("Data Loaded")

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

print("model created")

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by back propagation
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)

print("testing starts")

correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):

        img = images[i].view(1, 784)

        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])

        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]

        if true_label == pred_label:
            correct_count += 1

        all_count += 1

print("Number Of Images Tested =", all_count)

print("\nModel Accuracy =", (correct_count / all_count))

torch.save(model, 'model.pt')



