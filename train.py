import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json 
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.autograd import Variable 

data_dir = '/home/leon/machine_l/database/amap/'


AMAP_LABEL_NAMES = (
    'unblocked',
    'slow',
    'blocked',
    )

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        # img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    # if img.ndim == 2:
    #     # reshape (H, W) -> (1, H, W)
    #     return img[np.newaxis]
    # else:
    #     # transpose (H, W, C) -> (C, H, W)
    #     return img.transpose((2, 0, 1))

class AMAPDataset:
    def __init__(self, phase, transform=None):
        self.data_dir = data_dir
        self.label_names = AMAP_LABEL_NAMES
        self.phase = phase
        img_dir = ''
        json_file = ''
        
        self.ids = os.listdir(self.data_dir + img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):

        id_ = self.ids[i]
        # image = []
        # label = 0
        if self.phase == 'train':
            img_dir = 'amap_traffic_train_0712'
            json_file = "amap_traffic_annotations_train.json"
        elif self.phase == 'valid':
            img_dir = 'amap_traffic_test_0712'
            json_file = "amap_traffic_annotations_test.json"

        with open(self.data_dir+json_file,"r") as f:
            content=f.read()
        content=json.loads(content)
        cid = content['annotations'][i]
    
        # if id_ == cid['id']:      
        img_file = os.path.join(self.data_dir, img_dir, cid['id'], cid['key_frame'])
        # print(img_file)
        img = Image.open(img_file).convert('RGB')
        # img = read_image(img_file, color=True)
        img = self.transform(img)
        label = (cid['status'])+1
        return img, label
    __getitem__ = get_example


# train_directory = os.path.join(dataset, 'train')
# valid_directory = os.path.join(dataset, 'valid')
 
batch_size = 32
num_classes = 3
 
# data = {
#     'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
#     'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']) 
# }


# data_loaders = {'train': dataloader, 'val': test_dataloader}   
 
# train_data_size = len(data['train'])
# valid_data_size = len(data['valid'])
 
# train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
# valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)


train_dataset = AMAPDataset(phase = 'train',transform = image_transforms['train'])
valid_dataset = AMAPDataset(phase = 'valid',transform = image_transforms['valid'])
#设置batch size = 64

train_data_size = len(train_dataset)
valid_data_size = len(valid_dataset)

train_data = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
valid_data = DataLoader(dataset=valid_dataset)
 

resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 3),
    nn.LogSoftmax(dim=1)
)

resnet50 = resnet50.to('cuda:0')


loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())

def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0
 
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
 
        model.train()
 
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
 
        for i, (inputs, labels) in enumerate(train_data):
            # inputs = inputs.to(device)
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            # labels = labels.to(device)
            # print ('inputs shape and label shape', torch.Size(inputs), torch.Size(lables))
 
            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
 
            outputs = model(inputs)
 
            loss = loss_function(outputs, labels)
 
            loss.backward()
 
            optimizer.step()
            print(inputs.size(0))
 
            train_loss += loss.item() * inputs.size(0)
 
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
 
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
            train_acc += acc.item() * inputs.size(0)
 
        with torch.no_grad():
            model.eval()
 
            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                outputs = model(inputs)
 
                loss = loss_function(outputs, labels)
 
                valid_loss += loss.item() * inputs.size(0)
 
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
 
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
                valid_acc += acc.item() * inputs.size(0)
 
        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size
 
        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size
 
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
 
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
 
        epoch_end = time.time()
 
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_valid_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
 
        torch.save(model, 'models/'+dataset+'_model_'+str(epoch+1)+'.pt')
    return model, history

num_epochs = 3
trained_model, history = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
torch.save(history, 'models/'+dataset+'_history.pt')
 
history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig(dataset+'_loss_curve.png')
plt.show()
 
plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(dataset+'_accuracy_curve.png')
plt.show()
