import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.autograd import Variable


import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder

import numpy as np
import time
import copy
import os

from net import fine_tune_model

from tensorboardX import SummaryWriter


def train_model(data_loader, model, criterion, optimizer, scheduler, num_epochs=15):
    since_time = time.time()

    best_model = model
    best_acc = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # collect data info
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()
        if ((epoch + 1) % 3 == 0):
            save_torch_model(model, 'checkpoints/' + str(epoch) + '.pth')

    time_elapsed = time.time() - since_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


def save_torch_model(model, name):
    torch.save(model.state_dict(), name)


train_data = ImageFolder(root='train_data', transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]))

#print(train_data.class_to_idx)
    
VALID_RATIO = 0.8
BATCH_SIZE = 256

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

dataloaders = {
    'train':data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True),
    'valid':data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
}

writer = SummaryWriter()

os.makedirs('checkpoints', exist_ok=True)

model = fine_tune_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(dataloaders, model, criterion, optimizer, lr_scheduler)
save_torch_model(model, 'checkpoints/final.pth')