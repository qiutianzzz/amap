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

IMAGE_SIZE = 512 # 每张小图片的大小
IMAGE_ROW = 2 # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 1 # 图片间隔，也就是合并成一张图后，一共有几列
  

class COMPDataset:
    def __init__(self, img_dir, transform=None):
        self.data_dir = data_dir
        self.label_names = AMAP_LABEL_NAMES
        self.img_dir = img_dir
        self.ids = os.listdir(self.data_dir + self.img_dir)
        json_file = ''
        self.transform = transform

    def get_example(self, i):
        if self.img_dir == 'amap_traffic_train_0712':
            json_file = "amap_traffic_annotations_train.json"
        elif self.img_dir == 'amap_traffic_test_0712':
            json_file = "amap_traffic_annotations_test.json"
        
        # imgfolder = os.listdir(self.data_dir + self.img_dir + cid['id'])

        with open(self.data_dir+json_file,"r") as f:
            content=f.read()
        content=json.loads(content)
        cid = content['annotations'][i]
        IMAGE_PATH = self.data_dir + self.img_dir + '/'
        imgfiles = os.listdir(IMAGE_PATH + cid['id'])
        key_num, _ = content['annotations'][i]['key_frame'].split('.', 1)
        key_num = int(key_num)
        # img_frames = content['annotations'][i]['frames']
        img_nums = len(imgfiles)
        to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
        if img_nums == 1:
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGE_PATH + cid['id'] +'/'+ imgfiles[0]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
        
        elif img_nums == 2:
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGE_PATH + cid['id'] +'/'+ imgfiles[y-1]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
        
        elif img_nums == 3:
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGE_PATH + cid['id'] +'/'+ imgfiles[y]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
        
        else:
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGE_PATH  + cid['id'] +'/'+ imgfiles[key_num +y - 3]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
        img = self.transform(to_image)
        label = (cid['status'])+1
        return img, label

    def __len__(self):
        return len(self.ids)

    __getitem__ = get_example


class AMAPDataset:
    def __init__(self, img_dir, transform=None):
        self.data_dir = data_dir
        self.label_names = AMAP_LABEL_NAMES
        self.img_dir = img_dir
        self.ids = os.listdir(self.data_dir + self.img_dir)
        json_file = ''
        self.transform = transform

    def get_example(self, i):
        
        # image = []
        # label = 0
        if self.img_dir == 'amap_traffic_train_0712':
            json_file = "amap_traffic_annotations_train.json"
        elif self.img_dir == 'amap_traffic_test_0712':
            json_file = "amap_traffic_annotations_test.json"

        
        id_ = self.ids[i]
        with open(self.data_dir+json_file,"r") as f:
            content=f.read()
        content=json.loads(content)
        cid = content['annotations'][i]
    
        # if id_ == cid['id']:      
        img_file = os.path.join(self.data_dir, self.img_dir, cid['id'], cid['key_frame'])
        # print(img_file)
        img = Image.open(img_file).convert('RGB')
        # img = read_image(img_file, color=True)
        img = self.transform(img)
        label = (cid['status'])+1
        return img, label

    def __len__(self):
        return len(self.ids)
    __getitem__ = get_example

 
batch_size = 32
num_classes = 3

# train_dataset = AMAPDataset(img_dir = 'amap_traffic_train_0712',transform = image_transforms['train'])
# valid_dataset = AMAPDataset(img_dir = 'amap_traffic_test_0712',transform = image_transforms['valid'])
train_dataset = COMPDataset(img_dir = 'amap_traffic_train_0712',transform = image_transforms['train'])
valid_dataset = COMPDataset(img_dir = 'amap_traffic_test_0712',transform = image_transforms['valid'])
#设置batch size = 64
print(train_dataset.__len__())

train_data_size = len(train_dataset)
valid_data_size = len(valid_dataset)

train_data = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
valid_data = DataLoader(dataset=valid_dataset)
print(train_data.__len__())

resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.LogSoftmax(dim=1)
)

resnet50 = resnet50.to('cuda:0')


loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())

def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    results = []
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
            print(i)
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
            # print(inputs.size(0))
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
                # print ("return:, predictions", ret, predictions)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                # print ("correct_counts", correct_counts)
                if epoch == (epochs-1):
                    results.append(predictions.item()-1)
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
        if epoch % 10 == 1:
            torch.save(model, 'checkpoints/'+'_model_'+str(epoch+1)+'.pt')
    # print ('test_predictions', results)
    return model, history, results

num_epochs = 25
trained_model, history, results = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
torch.save(history, 'checkpoints/'+'_history.pt')
 
history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('_loss_curve.png')
# plt.show()
 
plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('_accuracy_curve.png')
# plt.show()

json_path = data_dir+"amap_traffic_annotations_test.json"
out_path = data_dir+"amap_traffic_annotations_test_result.json"

# result 是你的结果, key是id, value是status
with open(json_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
    json_dict = json.load(f)
    data_arr = json_dict["annotations"]  
    new_data_arr = [] 
    for data in data_arr:
        id_ = data["id"]
        id_ = int(id_)-1
        print(id_)
        data["status"] = int(results[id_])
        new_data_arr.append(data)
    json_dict["annotations"] = new_data_arr
    json.dump(json_dict, w)