import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_root=None):
        self.data_root = data_root
        self.data_index = self.build_index(self.data_root)

    def build_index(self, data_root):
        return [None, None]

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        sample = self.data_index[idx]
        
        # load data, NOTE: modify by cv2.imread(...)
        image = torch.rand(3, 240, 320)
        label = torch.rand(1, 240, 320)
        return dict(images=image, labels=label)


class MyModel(nn.Module):
    def __init__(self, num_outputs):
        super(MyModel, self).__init__()
        self.features = nn.Conv2d(3, num_outputs, 3, 1, 1)

    def forward(self, x):
        assert len(x.shape) == 4, x.shape
        return self.features(x)


def train(epoch, dataloader, model, criterion, optimizer, device):
    model.train()
    for i_batch, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)
        labels = sample_batched['labels'].to(device)
        
        output = model(images)
        
        loss = criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('Sample {0}/{1} Loss: {2}' \
              .format(i_batch, epoch, loss.item()))

                
def validation(epoch, dataloader, model, criterion, device):
    model.eval()
    total_losses = []
    for i_batch, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)
        labels = sample_batched['labels'].to(device)
        
        output = model(images)
        
        loss = criterion(output, labels)
        
        print('Sample {0}/{1} Loss: {2}' \
              .format(i_batch, epoch, loss.item()))

        total_losses.append(loss)
    mean_loss = torch.stack(total_losses).mean()
    print('Mean Loss: {}'.format(mean_loss.item()))
        
        
def main():
        device = torch.device('cpu')
        model = MyModel(num_outputs=1).to(device)
        dataset_train = MyDataset()
        dataset_val = MyDataset()

        dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        
        num_epochs = 10
        for epoch in range(num_epochs):
            train(epoch, dataloader_train, model, criterion, optimizer, device)
        
            with torch.no_grad():
                validation(epoch, dataloader_val, model, criterion, device)

        
if __name__ == '__main__':
    main()
