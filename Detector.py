from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
import torch
import time


# trying to change 
class Detector:
    def __init__(self, interested_classes):
        self.cfg = get_cfg()

        print("===== self.cfg =====")
        print(self.cfg)
        print("====================")

        # Load model config and pretrained model
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

        self.classes_dict = {}
        self.construct_class_dict()
        self.interested_classes = interested_classes
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


    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        cv2.imshow("Before network", image)
        # cv2.waitKey(0)

        predictions = self.predictor(image)

        print(predictions)

        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)

        return predictions


    def onVideo(self, videoPath, videoName, outputPath):
        video = cv2.VideoCapture(videoPath)
        name, _ = videoName.split(".mp4")  # .mp4 or .MOV
        output_fps = 1
        out = cv2.VideoWriter(outputPath+'{}_out.mp4'.format(name), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (1280,720))

        fps = video.get(cv2.CAP_PROP_FPS)

        if (video.isOpened()==False):
            print("Error opening the file...")
            return
        
        success, frame = video.read()
        frame_id = 0
        print("processing frames...")

        while success:
            if (frame_id+1) % (fps // output_fps) == 0:
                predictions = self.predictor(frame)

                viz = Visualizer(frame[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                
                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

                output_im = output.get_image()[:,:,::-1]
                out.write(output_im)
                

            success, frame = video.read()
            frame_id = (frame_id+1) % fps

        video.release()
        out.release()        
        cv2.destroyAllWindows()
        print("Done!")
    

    def onVideoNoOutput(self, videoPath, videoName, outputPath, driver_state, output_fps=5):
        video = cv2.VideoCapture(videoPath)
        name, _ = videoName.split(".mp4")  # .mp4 or .MOV
        out = cv2.VideoWriter(outputPath+'{}_out.mp4'.format(name), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (1280,720))

        fps = video.get(cv2.CAP_PROP_FPS)

        if (video.isOpened()==False):
            print("Error opening the file...")
            return
        
        success, frame = video.read()
        frame_id = 0
        print("processing frames...")
        state_index = 0

        while success:
            if (frame_id+1) % (fps // output_fps) == 0:

                start = time.time()

                outputs = self.predictor(frame)
                # import pdb
                # pdb.set_trace()
                # t_image = image[np.newaxis, :]
                # print(t_image.shape)
                # outputs = self.predictor(t_image)
                pred_classes = outputs["instances"].pred_classes
                pred_boxes = outputs["instances"].pred_boxes
                # print(pred_classes)
                # print(pred_boxes)

                pred_classes_list = pred_classes.tolist()

                indx_to_remove = []
                for i, num in enumerate(pred_classes_list):
                    if num not in self.interested_classes_num:
                        indx_to_remove.append(i)
                # print(indx_to_remove)

                pred_classes = np.delete(pred_classes.cpu().numpy(), indx_to_remove)
                pred_boxes = np.delete(pred_boxes.tensor.cpu().numpy(), indx_to_remove, axis=0)

                if (pred_classes):
                    print("#RoadView: detected pred_classes {}".format(pred_classes))
                    print("#RoadView: state_index: {}".format(state_index))
                    state = driver_state.states[state_index]
                    state.has_stop_sign = True 
                else:
                    print ("#RoadView: No pred_classes ..... ")
                
                print ("#RoadView: state_index is {}".format(state_index))

                end = time.time()

                print("time diff is {}".format(end - start))

            success, frame = video.read()
            frame_id = (frame_id+1) % fps
            state_index = state_index + 1

        video.release()
        out.release()        
        cv2.destroyAllWindows()
        print("Done!")
    

    def filtered_outputs(self, imagePath):
        image = cv2.imread(imagePath)
        # import pdb
        # pdb.set_trace()
        # t_image = image[np.newaxis, :]
        # print(t_image.shape)
        # outputs = self.predictor(t_image)
        outputs = self.predictor(image)
        pred_classes = outputs["instances"].pred_classes
        pred_boxes = outputs["instances"].pred_boxes
        # print(pred_classes)
        # print(pred_boxes)

        pred_classes_list = pred_classes.tolist()

        indx_to_remove = []
        for i, num in enumerate(pred_classes_list):
            if num not in self.interested_classes_num:
                indx_to_remove.append(i)
        # print(indx_to_remove)

        pred_classes = np.delete(pred_classes.cpu().numpy(), indx_to_remove)
        pred_boxes = np.delete(pred_boxes.tensor.cpu().numpy(), indx_to_remove, axis=0)
        # print(pred_classes)
        # print(pred_boxes)

        # for data in pred_classes:
        #     num = data.item()
        #     print(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[num])
        
        return pred_classes, pred_boxes
        


