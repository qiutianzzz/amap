import PIL.Image as Image
import os
from torchvision import datasets, models, transforms
import json
  
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
IMAGE_COLUMN = 2 # 图片间隔，也就是合并成一张图后，一共有几列
  

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
        # key_num, _ = content['annotations'][i]['key_frame'].split('.', 1)
        # key_num = int(key_num)
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
                    from_image = Image.open(IMAGE_PATH  + cid['id'] +'/'+ imgfiles[img_nums - (IMAGE_COLUMN *(y-1) + x)]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
        # img = self.transform(to_image)
        img = to_image
        label = (cid['status'])+1
        return img, label

    def __len__(self):
        return len(self.ids)

    __getitem__ = get_example

data = COMPDataset('amap_traffic_train_0712', transform = None)
im, _ = data.get_example(10)
# unloader = transforms.ToPILImage()
# image = im.cpu().clone()  # clone the tensor
# image = im
# image = image.squeeze(0)  # remove the fake batch dimension
# image = unloader(image)
# image.save('example.jpg')
# im = Image.open('example.jpg')
im.show()