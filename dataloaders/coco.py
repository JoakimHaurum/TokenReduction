import os
import torch
from torchvision import datasets as datasets
from pycocotools.coco import COCO 
from PIL import Image

class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        
        if train:
            self.dataPath = os.path.join(root, "train2014")
            self.annFile = os.path.join(root, 'annotations', 'instances_train2014.json')
        else:
            self.dataPath = os.path.join(root, "val2014")
            self.annFile = os.path.join(root, 'annotations', 'instances_val2014.json')
        
        self.coco = COCO(self.annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
            
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.dataPath, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

