import torch
import torch.nn as nn
import torchvision
import numpy as np
import copy
from torch.autograd import Variable
from torch.utils import model_zoo
from utils.config import opt
from data.amapdata import Dataset, TestDataset, inverse_normalize
from torch.utils import data as data_
 
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
 
__all__ = ['ResNet50', 'ResNet101','ResNet152']
 
def Conv1(in_planes, places, stride=2):
  return nn.Sequential(
    nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
    nn.BatchNorm2d(places),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  )
 
class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
      super(Bottleneck,self).__init__()
      self.expansion = expansion
      self.downsampling = downsampling
  
      self.bottleneck = nn.Sequential(
        nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(places*self.expansion),
      )
  
      if self.downsampling:
        self.downsample = nn.Sequential(
          nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(places*self.expansion)
        )
      self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
      residual = x
      out = self.bottleneck(x)
  
      if self.downsampling:
        residual = self.downsample(x)
  
      out += residual
      out = self.relu(out)
      return out
 

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=3, expansion = 4):
      super(ResNet,self).__init__()
      self.expansion = expansion
  
      self.conv1 = Conv1(in_planes = 3, places= 64)
  
      self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
      self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
      self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
      self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)
  
      self.avgpool = nn.AvgPool2d(7, stride=1)
      self.fc = nn.Linear(2048,num_classes)
  
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
  
    def make_layer(self, in_places, places, block, stride):
      layers = []
      layers.append(Bottleneck(in_places, places,stride, downsampling =True))
      for i in range(1, block):
        layers.append(Bottleneck(places*self.expansion, places))
  
      return nn.Sequential(*layers)

    def forward(self, x):
      x = self.conv1(x)
  
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
  
      x = self.avgpool(x)
      x = x.view(x.size(0), -1)
      x = self.fc(x)
      # add the softmax classification 
      return x

    # def get_optimizer(self):
    #     """
    #     return optimizer, It could be overwriten if you want to specify 
    #     special optimizer
    #     """
    #     lr = opt.lr
    #     params = []
    #     for key, value in dict(self.named_parameters()).items():
    #         if value.requires_grad:
    #             if 'bias' in key:
    #                 params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
    #             else:
    #                 params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
    #     if opt.use_adam:
    #         self.optimizer = t.optim.Adam(params)
    #     else:
    #         self.optimizer = t.optim.SGD(params, momentum=0.9)
    #     return self.optimizer

    # def scale_lr(self, decay=0.1):
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] *= decay
    #     return self.optimizer


def ResNet50(pretrained = False):
    model = ResNet([3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
 
def ResNet101(pretrained = False):
    model = ResNet([3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
 
def ResNet152(pretrained = False):
    model = ResNet([3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def train_model(model, criterion, optimizer, scheduler, **kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print(dataset)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    data_loaders = {'train': dataloader, 'val': test_dataloader}                        
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}
     
    best_model_wts = copy.deepcopy(model.state_dict())
    best_class_acc = 0.0
    best_specy_acc = 0.0
    best_acc = 0.0
    best_loss = 10000
    
    #epoch循环训练
    for epoch in range(opt.epoch):
        print('epoch {}/{}'.format(epoch,opt.epoch - 1))
        print('-*' * 10)
        
        # 每个epoch都有train(训练)和val(测试)两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            corrects_classes = 0
            corrects_species = 0
            
            for idx, (img, label, scale) in enumerate(data_loaders[phase]):
                
                #将数据存在gpu上
                inputs = Variable(img.cuda())
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels_classes = Variable(label.cuda())
                labels_classes = labels_classes.type(torch.cuda.FloatTensor)
                # labels_species = Variable(data['species'].cuda())
                optimizer.zero_grad()
                
                #训练阶段
                with torch.set_grad_enabled(phase == 'train'):
                    x_classes = model(inputs)                    
                    _, preds_classes = torch.max(x_classes, 1)      
                    # _, preds_species = torch.max(x_species, 1)
                    #计算训练误差
                    loss = criterion(x_classes, labels_classes)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)          
                corrects_classes += torch.sum(preds_classes == labels_classes)
                # corrects_species += torch.sum(preds_species == labels_species)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)
            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            # epoch_acc_species = corrects_species.double() / len(data_loaders[phase].dataset)

            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            # Accuracy_list_species[phase].append(100 * epoch_acc_species)
            print('{} Loss: {:.4f}  Acc_classes: {:.2%}'
                  .format(phase, epoch_loss,epoch_acc_classes))

            #测试阶段
            if phase == 'val':
                #如果当前epoch下的准确率总体提高或者误差下降，则认为当下的模型最优
                if epoch_acc_classes > best_acc or epoch_loss < best_loss:
                    best_acc_classes = epoch_acc_classes
                    # best_acc_species = epoch_acc_species
                    best_acc = best_acc_classes
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('Best_model:  classes Acc: {:.2%}'
                          .format(best_acc_classes))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best_model:  classes Acc: {:.2%}'
          .format(best_acc_classes))
    return model, Loss_list,Accuracy_list_classes

if __name__ == '__main__':
    
  net = ResNet50()
  network = net.cuda()
  optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
  # criterion = nn.CrossEntropyLoss()
  criterion = labels.squeeze(1)
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs
  model, Loss_list, Accuracy_list_classes = train_model(network, criterion, optimizer, exp_lr_scheduler)
# if __name__=='__main__':
  #model = torchvision.models.resnet50()
  # model = ResNet50()
#   print(model)
 
# change the input to the amap images 
  # input = torch.randn(1, 3, 224, 224)
  # out = model(input)
  # print(out)

# where is the optimizer? feedback? backpropogation?
