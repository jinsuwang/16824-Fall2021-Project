import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import cv2
import numpy as np
import torch

# trying to change 
class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        # Load model config and model weights
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"
        self.classes = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes

        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load('saved_models/model_final_f6e8b1.pkl') # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
        self.model.train(False) # inference mode
        print("Done initializing")


        self.classes_dict = {}
        
        self.construct_class_dict()
        
        self.interested_classes = ["person", "car", "truck", "stop sign"]

        self.interested_classes_num = self.construct_interested_cls_num()


    def construct_class_dict(self):
        for i, name in enumerate(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes):
            self.classes_dict[name] = i
        
    def construct_interested_cls_num(self):
        intr_cls_num = []
        for name in self.interested_classes:
            num = self.classes_dict[name]
            intr_cls_num.append(num)
        return intr_cls_num



    def filtered_outputs(self, pred_classes, pred_boxes):
        pred_classes_list = pred_classes.tolist()

        indx_to_remove = []
        for i, num in enumerate(pred_classes_list):
            if num not in self.interested_classes_num:
                indx_to_remove.append(i)

        # convert to numpy to use np.delete
        pred_classes = np.delete(pred_classes.cpu().detach().numpy(), indx_to_remove)
        pred_boxes = np.delete(pred_boxes.cpu().detach().numpy(), indx_to_remove, axis=0)
        
        # convert back to tensor
        pred_classes = torch.tensor(pred_classes).cuda()
        pred_boxes = torch.tensor(pred_boxes).cuda()
        
        # # Print outs for debugging
        # print(indx_to_remove)
        # print(pred_classes)
        # print(pred_boxes)
        # for data in pred_classes:
        #     num = data.item()
            # print(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[num])
        
        return pred_classes, pred_boxes



    def single_inference(self, image_path, filter=False):
        """Takes in a path to an image and returns tensors for the classes (indices)
            in the image, and a tensor for the bounding boxes.

        Args:
            image_path (string): path to a single images
            filter (boolean): whether to filter out unwanted classes

        Returns:
            pred_classes: tensor with the indices of identified classes
            pred_boxes: tensor with the bounding box coordinates (top left and bottom right) for each instance identified
        """

        img = cv2.imread(image_path)
        img = np.transpose(img,(2,0,1))
        img_tensor = torch.from_numpy(img)
        inputs = [{"image":img_tensor}] # inputs is ready

        outputs = self.model(inputs)

        pred_classes = outputs[0]["instances"].pred_classes.detach()
        pred_boxes = outputs[0]["instances"].pred_boxes.tensor.detach()
        
        if filter:
            pred_classes, pred_boxes = self.filtered_outputs(pred_classes, pred_boxes)

        return pred_classes, pred_boxes

    
    def batch_inference_frompaths(self, list_of_paths, filter=False):
        """Takes in a list of paths to images and returns a list of tensors for the classes (indices)
            in the images, and a list of tensors for the bounding boxes.

        Args:
            image_path (list of string): list of paths to images
            filter (boolean): whether to filter out unwanted classes

        Returns:
            list_of_pred_classes: list of tensors with the indices of identified classes.
                                If filter is True, then it is possible each of the tensors will be completely empty.
            list_of_pred_boxes: list of tensors with the bounding box coordinates (top left and bottom right) for each instance identified
                                If filter is True, then it is possible each of the tensors will be completely empty.
        """
        inputs = []
        for image_path in list_of_paths:
            img = cv2.imread(image_path)
            img = np.transpose(img,(2,0,1))
            img_tensor = torch.from_numpy(img)
            inputs.append({"image":img_tensor})

        outputs = self.model(inputs)

        list_of_pred_classes = []
        list_of_pred_boxes = []
        for i in range(0, len(outputs)):
            pred_classes = outputs[i]["instances"].pred_classes.detach()
            pred_boxes = outputs[i]["instances"].pred_boxes.tensor.detach()
            
            if filter:
                pred_classes, pred_boxes = self.filtered_outputs(pred_classes, pred_boxes)

            list_of_pred_classes.append(pred_classes)
            list_of_pred_boxes.append(pred_boxes)


        return list_of_pred_classes, list_of_pred_boxes


    def batch_inference_fromtensors(self, stacked_tensors, filter=False):
        """Takes in a batched tensor of images of shape (batch_size, C, H, W) 
            and returns a list of tensors for the classes (indices)
            in the images, and a list of tensors for the bounding boxes.

        Args:
            stacked_tensors (tensor): tensor of shape (batch_size, C, H, W) that contains images (e.g. from a dataloader)
            filter (boolean): whether to filter out unwanted classes

        Returns:
            list_of_pred_classes: list of tensors with the indices of identified classes.
                                If filter is True, then it is possible each of the tensors will be completely empty.
            list_of_pred_boxes: list of tensors with the bounding box coordinates (top left and bottom right) for each instance identified
                                If filter is True, then it is possible each of the tensors will be completely empty.
        """
        batch_size = stacked_tensors.size()[0]
        inputs = []
        for i in range(0, batch_size):
            img_tensor = stacked_tensors[i, :]
            img_tensor = img_tensor.squeeze()
            inputs.append({"image":img_tensor})

        outputs = self.model(inputs)

        list_of_pred_classes = []
        list_of_pred_boxes = []
        for i in range(0, len(outputs)):
            pred_classes = outputs[i]["instances"].pred_classes.detach()
            pred_boxes = outputs[i]["instances"].pred_boxes.tensor.detach()
            
            if filter:
                pred_classes, pred_boxes = self.filtered_outputs(pred_classes, pred_boxes)

            list_of_pred_classes.append(pred_classes)
            list_of_pred_boxes.append(pred_boxes)


        return list_of_pred_classes, list_of_pred_boxes

