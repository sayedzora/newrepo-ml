
import os
import yaml
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


import transforms as T



class_dict = {
'cup or mug': 0,
'bird': 1,
'hat with a wide brim': 2,
'person': 3,
'dog': 4,
'lizard': 5,
'sheep': 6,
'wine bottle': 7,
'bowl': 8,
'airplane': 9,
'domestic cat': 10,
'car': 11,
'porcupine': 12,
'bear': 13,
'tape player': 14,
'ray': 15,
'laptop': 16,
'zebra': 17,
'computer keyboard': 18,
'pitcher': 19,
'artichoke': 20,
'tv or monitor': 21,
'table': 22,
'chair': 23,
'helmet': 24,
'traffic light': 25,
'red panda': 26,
'sunglasses': 27,
'lamp': 28,
'bicycle': 29,
'backpack': 30,
'mushroom': 31,
'fox': 32,
'otter': 33,
'guitar': 34,
'microphone': 35,
'strawberry': 36,
'stove': 37,
'violin': 38,
'bookshelf': 39,
'sofa': 40,
'bell pepper': 41,
'bagel': 42,
'lemon': 43,
'orange': 44,
'bench': 45,
'piano': 46,
'flower pot': 47,
'butterfly': 48,
'purse': 49,
'pomegranate': 50,
'train': 51,
'drum': 52,
'hippopotamus': 53,
'ski': 54,
'ladybug': 55,
'banana': 56,
'monkey': 57,
'bus': 58,
'miniskirt': 59,
'camel': 60,
'cream': 61,
'lobster': 62,
'seal': 63,
'horse': 64,
'cart': 65,
'elephant': 66,
'snake': 67,
'fig': 68,
'watercraft': 69,
'apple': 70,
'antelope': 71,
'cattle': 72,
'whale': 73,
'coffee maker': 74,
'baby bed': 75,
'frog': 76,
'bathing cap': 77,
'crutch': 78,
'koala bear': 79,
'tie': 80,
'dumbbell': 81,
'tiger': 82,
'dragonfly': 83,
'goldfish': 84,
'cucumber': 85,
'turtle': 86,
'harp': 87,
'jellyfish': 88,
'swine': 89,
'pretzel': 90,
'motorcycle': 91,
'beaker': 92,
'rabbit': 93,
'nail': 94,
'axe': 95,
'salt or pepper shaker': 96,
'croquet ball': 97,
'skunk': 98,
'starfish': 99
}

inv_class_dict = {value: key for key, value in class_dict.items()}

from dataset import UnlabeledDataset, LabeledDataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


@torch.no_grad()
def main(load = None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(annotations_path):
        os.mkdir(annotations_path)

    num_classes = 100

    valid_dataset = UnlabeledDataset(root=image_dir, transform=get_transform(train=False),  send_target = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    model = get_model(num_classes)
    if load:
        checkpoint = torch.load(load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model Loaded")

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Annotate:'

    # coco = get_coco_api_from_dataset(data_loader.dataset)
    # iou_types = _get_iou_types(model)
    # coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(valid_loader, 100, header):
        images = list(img.to(device) for img in images)

        outputs = model(images)
        for output, target in zip(outputs, targets):
            filter = output['scores'] > 0.5
            boxes = output['boxes'][filter]
            labels = output['labels'][filter]
            idx = target["image_id"].item()
            image_size = target["image_size"]

            # print(boxes, labels, idx, image_size)
            data = {}
            data["bboxes"] = boxes.int().tolist()
            data["image_size"] = [int(image_size[0]), int(image_size[1])]
            data["labels"] =[ inv_class_dict[class_idx] for class_idx in labels.int().tolist()]

            print(data)

            with open(os.path.join(annotations_path, f'{idx}.yml'), 'w') as f:
                yaml.dump(data, f, sort_keys=False, default_flow_style=False)



if __name__ == "__main__":

    
    weight_path = "prev_epoch_0_weight.pth"
    image_dir = r"/unlabeled"
    annotations_path = 'unlabeled_anotations'
    main(load = "epoch_21_weight.pth")
