# 5. 성능 향상을 위한 방법

이번 장에서는 2-stage-method인 Faster R-CNN으로 객체 탐지를 해보도록 하겠습니다.

실험에 앞서 Google Colab에서는 랜덤 GPU를 할당하고 있기 때문에 메모리 부족현상이 일어날 수 있습니다.

먼저 GPU를 확인 후에 메모리가 충분할 경우 실험을 하시길 권장합니다.
(런타임을 초기화할 경우 새로운 GPU를 할당받으실 수 있습니다)


```python
import torch

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
```

    There are 1 GPU(s) available.
    We will use the GPU: Tesla T4
    

##5.1 데이터 불러오기
2.2절에서 보았듯이 가짜연구소 API를 활용하여 데이터를 다운받고 압축 파일을 풀어줍니다.



```python
!git clone https://github.com/Pseudo-Lab/Tutorial-Book-DataLoader
!python Tutorial-Book-DataLoader/PL_data_loader.py --data FaceMaskDetection
!unzip -q Face\ Mask\ Detection.zip
```

    Cloning into 'Tutorial-Book-DataLoader'...
    remote: Enumerating objects: 9, done.[K
    remote: Counting objects: 100% (9/9), done.[K
    remote: Compressing objects: 100% (8/8), done.[K
    remote: Total 9 (delta 1), reused 2 (delta 0), pack-reused 0[K
    Unpacking objects: 100% (9/9), done.
    Face Mask Detection.zip is done!
    

## 5.2 데이터 분리
먼저 이전장에서 사용했던 패키지들과 모델에 필요한 패키지를 불러오겠습니다.

`torchvision`은 이미지 처리를 하기 위해 사용되며 데이터셋에 관한 패키지와 모델에 관한 패키지가 내장되어 있습니다.



```python
import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from PIL import Image
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
```

그리고 3.4절과 같이 총 853장의 images와 annotations을 학습 데이터와 시험 데이터로 분리합니다.

test 폴더를 만들어 8:2의 비율로 시험 데이터를 따로 옮겨줍니다.


```python
print(len(os.listdir('annotations')))
print(len(os.listdir('images')))
```

    853
    853
    


```python
!mkdir test_images
!mkdir test_annotations
```


```python
import shutil

for img in sorted(os.listdir('images'))[-170:]:
    shutil.move('images/'+img, 'test_images/'+img)

for annot in sorted(os.listdir('annotations'))[-170:]:
    shutil.move('annotations/'+annot, 'test_annotations/'+annot)
```

위의 코드에서 shutil 패키지는 파일과 폴더 관련한 이동, 복사 작업을 할 때 사용합니다.


```python
print(len(os.listdir('annotations')))
print(len(os.listdir('images')))
print(len(os.listdir('test_annotations')))
print(len(os.listdir('test_images')))
```

    683
    683
    170
    170
    

## 5.3 데이터셋 클래스 정의
2.3절과 마찬가지로 바운딩 박스를 위한 함수들을 정의해줍니다.


```python
def generate_box(obj):
    
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0

def generate_target(image_id, file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'html.parser')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        
        return target
```

또한 4.3절처럼 데이터셋 클래스와 데이터 로더를 정의해줍니다.

데이터셋은  `torch.utils.data.DataLoader` 함수를 통해 배치 사이즈는 4로 지정하여 불러오겠습니다.

배치 사이즈는 개인의 메모리 크기에 따라 자유롭게 설정하면 됩니다. 


```python
class MaskDataset(object):
    def __init__(self, transforms, path):
        '''
        path: path to train folder or test folder
        '''
        # transform module과 img path 경로를 정의
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))


    def __getitem__(self, idx): #special method
        # load images ad masks
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + 'xml'
        img_path = os.path.join(self.path, file_image)
        
        if 'test' in self.path:
            label_path = os.path.join("test_annotations/", file_label)
        else:
            label_path = os.path.join("annotations/", file_label)

        img = Image.open(img_path).convert("RGB")
        #Generate Label
        target = generate_target(idx, label_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self): # len() 적용가능케 함, special method
        return len(self.imgs)

data_transform = transforms.Compose([  # transforms.Compose : list 내의 작업을 연달아 할 수 있게 호출하는 클래스
        transforms.ToTensor() # ToTensor : numpy 이미지에서 torch 이미지로 변경
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

dataset = MaskDataset(data_transform, 'images/')
test_dataset = MaskDataset(data_transform, 'test_images/')

data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)
```

##5.4 모델 불러오기
`torchvision.models.detection`에서는 Faster R-CNN API(`torchvision.models.detection.fasterrcnn_resnet50_fpn`)를 제공하고 있어 쉽게 구현이 가능합니다.

이는 COCO 데이터셋을 ResNet50으로 pre-trained한 모델을 제공하고 있으며, `pretrained=True/False`로 설정할 수 있습니다.

그리고 이후 모델을 불러올 때 `num_classes`에 원하는 클래스 개수를 설정하고 모델을 사용하면 됩니다.

이렇게 간단하게 Faster R-CNN 모델을 실험해볼 수 있습니다.


```python
def get_model_instance_segmentation(num_classes):
  
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
```

##5.5 전이학습
그럼 Face Mask Detection에 적용해보도록 하겠습니다.

Face Mask Detection 데이터셋은 3개의 클래스로 이루어져 있기 때문에 `num_classes`를 3으로 설정한 후 모델을 불러옵니다.

GPU를 사용할 수 있는 환경이라면 device로 지정하여 불러온 모델을 GPU에 보내줍니다.


```python
model = get_model_instance_segmentation(3)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
model.to(device)
```

    Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    


    HBox(children=(FloatProgress(value=0.0, max=167502836.0), HTML(value='')))


    
    




    FasterRCNN(
      (transform): GeneralizedRCNNTransform(
          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          Resize(min_size=(800,), max_size=1333, mode='bilinear')
      )
      (backbone): BackboneWithFPN(
        (body): IntermediateLayerGetter(
          (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          (bn1): FrozenBatchNorm2d(64)
          (relu): ReLU(inplace=True)
          (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
          (layer1): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(64)
              (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(64)
              (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(256)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): FrozenBatchNorm2d(256)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(64)
              (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(64)
              (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(256)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(64)
              (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(64)
              (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(256)
              (relu): ReLU(inplace=True)
            )
          )
          (layer2): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(128)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(128)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(512)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): FrozenBatchNorm2d(512)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(128)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(128)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(512)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(128)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(128)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(512)
              (relu): ReLU(inplace=True)
            )
            (3): Bottleneck(
              (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(128)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(128)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(512)
              (relu): ReLU(inplace=True)
            )
          )
          (layer3): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): FrozenBatchNorm2d(1024)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024)
              (relu): ReLU(inplace=True)
            )
            (3): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024)
              (relu): ReLU(inplace=True)
            )
            (4): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024)
              (relu): ReLU(inplace=True)
            )
            (5): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(256)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(256)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(1024)
              (relu): ReLU(inplace=True)
            )
          )
          (layer4): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(512)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(512)
              (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(2048)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): FrozenBatchNorm2d(2048)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(512)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(512)
              (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(2048)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d(512)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d(512)
              (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d(2048)
              (relu): ReLU(inplace=True)
            )
          )
        )
        (fpn): FeaturePyramidNetwork(
          (inner_blocks): ModuleList(
            (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
            (3): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (layer_blocks): ModuleList(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (extra_blocks): LastLevelMaxPool()
        )
      )
      (rpn): RegionProposalNetwork(
        (anchor_generator): AnchorGenerator()
        (head): RPNHead(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
          (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (roi_heads): RoIHeads(
        (box_roi_pool): MultiScaleRoIAlign()
        (box_head): TwoMLPHead(
          (fc6): Linear(in_features=12544, out_features=1024, bias=True)
          (fc7): Linear(in_features=1024, out_features=1024, bias=True)
        )
        (box_predictor): FastRCNNPredictor(
          (cls_score): Linear(in_features=1024, out_features=3, bias=True)
          (bbox_pred): Linear(in_features=1024, out_features=12, bias=True)
        )
      )
    )



위의 출력되는 결과를 통해 Fastser R-CNN이 어떤 layer들로 구성되어 있는지 알 수 있습니다.

이 때, GPU 사용 가능 여부는 `torch.cuda.is_available()`를 통해 알 수 있습니다.


```python
torch.cuda.is_available()
```




    True



이제 모델이 만들어졌으니 학습을 해보겠습니다.

학습 횟수(`num_epochs`)는 30으로 지정하고, SGD 방법을 통해 최적화 시켜보겠습니다.

각 하이퍼 파라미터는 자유롭게 수정하여 사용할 수 있습니다.


```python
num_epochs = 30
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
```

이제 학습을 시켜보겠습니다.

위에서 생성한 data_loader에서 한 배치씩 순서대로 모델에 사용하며, 이후 loss 계산을 통해 최적화를 수행합니다.

지정한 에폭만큼 반복하고 각 에폭마다 loss를 출력하여 확인합니다.


```python
print('----------------------train start--------------------------')
for epoch in range(num_epochs):
    start = time.time()
    model.train()
    i = 0    
    epoch_loss = 0
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations) 
        losses = sum(loss for loss in loss_dict.values())        

        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        epoch_loss += losses
    print(f'epoch : {epoch+1}, Loss : {epoch_loss}, time : {time.time() - start}')
```

    ----------------------train start--------------------------
    epoch : 1, Loss : 57.07004928588867, time : 269.2427554130554
    epoch : 2, Loss : 39.43512725830078, time : 268.7257990837097
    epoch : 3, Loss : 32.92606735229492, time : 267.9897265434265
    epoch : 4, Loss : 28.937482833862305, time : 268.75047993659973
    epoch : 5, Loss : 28.95316505432129, time : 269.1901903152466
    epoch : 6, Loss : 26.549741744995117, time : 268.8914408683777
    epoch : 7, Loss : 23.126197814941406, time : 269.02998542785645
    epoch : 8, Loss : 24.639511108398438, time : 269.36951088905334
    epoch : 9, Loss : 22.880359649658203, time : 268.1531283855438
    epoch : 10, Loss : 19.464181900024414, time : 269.66372299194336
    epoch : 11, Loss : 19.465904235839844, time : 269.33947587013245
    epoch : 12, Loss : 19.406227111816406, time : 269.8699383735657
    epoch : 13, Loss : 17.55661964416504, time : 269.1940002441406
    epoch : 14, Loss : 18.008996963500977, time : 269.1736307144165
    epoch : 15, Loss : 17.95616912841797, time : 268.4821572303772
    epoch : 16, Loss : 15.526053428649902, time : 269.18503165245056
    epoch : 17, Loss : 14.976873397827148, time : 269.7834367752075
    epoch : 18, Loss : 16.75909996032715, time : 269.47766184806824
    epoch : 19, Loss : 16.092985153198242, time : 266.8096544742584
    epoch : 20, Loss : 15.812982559204102, time : 266.26549243927
    epoch : 21, Loss : 15.693418502807617, time : 266.25178718566895
    epoch : 22, Loss : 16.75595474243164, time : 266.33044385910034
    epoch : 23, Loss : 17.68772315979004, time : 266.0242862701416
    epoch : 24, Loss : 16.830270767211914, time : 265.3167517185211
    epoch : 25, Loss : 15.386274337768555, time : 264.8526291847229
    epoch : 26, Loss : 15.757489204406738, time : 265.6499161720276
    epoch : 27, Loss : 16.870962142944336, time : 265.70189142227173
    epoch : 28, Loss : 13.835110664367676, time : 264.8233985900879
    epoch : 29, Loss : 12.407930374145508, time : 264.04496908187866
    epoch : 30, Loss : 12.587990760803223, time : 264.01971983909607
    

학습시킨 가중치를 저장하고 싶다면, `torch.save`를 이용하여 저장해두고 나중에 언제든지 불러와 사용할 수 있습니다.


```python
torch.save(model.state_dict(),f'model_{num_epochs}.pt')
```


```python
model.load_state_dict(torch.load(f'model_{num_epochs}.pt'))
```




    <All keys matched successfully>



##5.6 예측
모델 학습이 끝났으면 잘 학습되었는지 예측 결과를 확인해보겠습니다.

우선 test_data_loader의 첫번째 배치에 대해서 결과를 출력해보았습니다.

아래와 같이 4장의 이미지(1배치)에 대해서 바운딩 박스의 좌표(boxes)와 클래스(labels), 점수(scores)가 예측됩니다.


```python
model.eval()
for imgs, annotations in test_data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

    model.eval()
    pred = model(imgs)
    print(pred)
    break
```

    [{'boxes': tensor([[242.3134,  68.6447, 340.8331, 190.1095]], device='cuda:0',
           grad_fn=<StackBackward>), 'labels': tensor([1], device='cuda:0'), 'scores': tensor([0.9971], device='cuda:0', grad_fn=<IndexBackward>)}, {'boxes': tensor([[377.1269, 123.7803, 400.0000, 156.6798],
            [304.5529, 187.4064, 324.6962, 209.4973],
            [185.9639, 153.3576, 206.7260, 182.1377],
            [ 78.9946, 141.0197, 107.9893, 172.7239]], device='cuda:0',
           grad_fn=<StackBackward>), 'labels': tensor([1, 1, 1, 1], device='cuda:0'), 'scores': tensor([0.9988, 0.9969, 0.9947, 0.9900], device='cuda:0',
           grad_fn=<IndexBackward>)}, {'boxes': tensor([[ 84.3238, 167.3977, 126.9322, 221.3025]], device='cuda:0',
           grad_fn=<StackBackward>), 'labels': tensor([1], device='cuda:0'), 'scores': tensor([0.9975], device='cuda:0', grad_fn=<IndexBackward>)}, {'boxes': tensor([], device='cuda:0', size=(0, 4), grad_fn=<StackBackward>), 'labels': tensor([], device='cuda:0', dtype=torch.int64), 'scores': tensor([], device='cuda:0', grad_fn=<IndexBackward>)}]
    

그러면 예측한 바운딩 박스를 그림으로 출력해보겠습니다.

아래 `plot_image` 함수를 만들어 이미지 위에 직사각형 모양의 바운딩 박스를 표시하였습니다.



```python
def plot_image(img_tensor, annotation):
    
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))
    
    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 0 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        
        elif annotation['labels'][idx] == 1 :
            
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
            
        else :
        
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='b',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
```

Prediction이 학습시킨 모델의 결과이고, Target이 실제 바운딩박스입니다.

Target과 똑같이 마스크를 쓴 사람 한 명을 제대로 예측하고 있습니다.


```python
print("Prediction : ", pred[0]['labels'])
plot_image(imgs[0], pred[0])
print("Target : ", annotations[0]['labels'])
plot_image(imgs[0], annotations[0])
```

    Prediction :  tensor([1], device='cuda:0')
    


![png](output_34_1.png)


    Target :  tensor([1], device='cuda:0')
    


![png](output_34_3.png)


이번엔 전체 시험 데이터에 대해서 예측 결과를 평가해보도록 하겠습니다. 예측 평가 지표로는 mAP를 사용합니다.


```python
def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i] # predict
        pred_boxes = output['boxes']
        pred_scores = output['scores']
        pred_labels = output['labels']

        true_positives = torch.zeros(pred_boxes.shape[0])   # 예측 객체 개수
 
        annotations = targets[sample_i]  # actual
        target_labels = annotations['labels'] if len(annotations) else []
        if len(annotations):    # len(annotations) = 3
            detected_boxes = []
            target_boxes = annotations['boxes']

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations): 
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)   # box_index : 실제 어떤 바운딩 박스랑 IoU 가 가장 높은지 index
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]  # 예측된거랑 실제랑 매핑해서 하나씩 index 채움
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics
```


```python
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
```


```python
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = torch.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = torch.unique(target_cls)   # 2가 거의 예측안됨

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = torch.cumsum(1 - tp[i],-1)
            tpc = torch.cumsum(tp[i],-1)

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = torch.tensor(np.array(p)), torch.tensor(np.array(r)), torch.tensor(np.array(ap))
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes
```


```python
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
```


```python
labels = []
sample_metrics = []  # List of tuples (TP, confs, pred)
for imgs, annotations in test_data_loader:
    imgs = list(img.to(device) for img in imgs)
    for t in annotations:
      labels += t['labels']
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    
    with torch.no_grad():
      model.eval()
      pred = model(imgs)
      sample_metrics += get_batch_statistics(pred, annotations, iou_threshold=0.5)
      
# Concatenate sample statistics
true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]  # 배치가 전부 합쳐짐
precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
mAP = torch.mean(AP)
print(f'mAP : {mAP}')
print(f'AP : {AP}')
```

    mAP : 0.2140198543853392
    AP : tensor([0.0000, 0.4726, 0.1695], dtype=torch.float64)
    


```python
# AP, mAP
```




    (tensor([0.0000, 0.4729, 0.2351], dtype=torch.float64),
     tensor(0.2360, dtype=torch.float64))




```python

```
