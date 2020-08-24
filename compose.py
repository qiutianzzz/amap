import PIL.Image as Image
import os
  
data_dir = 'C:\\AAA\\Add-Tech\\database\\amap\\' # 图片集地址

IMAGE_SIZE = 512 # 每张小图片的大小
IMAGE_ROW = 2 # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 2 # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'final.jpg' # 图片转换后的地址
  

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
        IMAGE_PATH = self.data_dir + self.img_dir
        imgfiles = os.listdir(IMAGE_PATH + cid['id'])
        # key_num, _ = content['annotations'][i]['key_frame'].split('.', 1)
        # key_num = int(key_num)
        # img_frames = content['annotations'][i]['frames']
        img_nums = len(imgfiles)
        to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
        if img_nums == 1:
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGE_PATH + imgfiles[0]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
        
        elif img_nums == 2:
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGE_PATH + imgfiles[y-1]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
        
        elif img_nums == 3:
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGE_PATH + imgfiles[y]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
        
        else:
            for y in range(1, IMAGE_ROW + 1):
                for x in range(1, IMAGE_COLUMN + 1):
                    from_image = Image.open(IMAGE_PATH + imgfiles[img_nums - (IMAGE_COLUMN *(y-1) + x-1)]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
        img = self.transform(to_image)
        label = (cid['status'])+1
        return img, label

    def __len__(self):
        return len(self.ids)

    __getitem__ = get_example
        


# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
    os.path.splitext(name)[1] == item]
  
# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
 raise ValueError("合成图片的参数和要求的数量不能匹配！")
  
# 定义图像拼接函数
def image_compose():
 to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
 # 循环遍历，把每张图片按顺序粘贴到对应位置上
 for y in range(1, IMAGE_ROW + 1):
  for x in range(1, IMAGE_COLUMN + 1):
   from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
    (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
   to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
 return to_image.save(IMAGE_SAVE_PATH) # 保存新图
image_compose() #调用函数
