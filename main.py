from Detector import *
from driver_state.driver_state import DriverState, State
from driver_state.evaluation_strategy import StopSignEvaluationStrategy
from label_reader import LabelReader
import argparse
import time
import warnings
import numpy as np
import torch
import math
import torchvision
from torchvision import transforms
import cv2
from dectect import AntiSpoofPredict
from pfld.pfld import PFLDInference, AuxiliaryNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument(
        '--driver_model_path',
        default="./checkpoint/snapshot/checkpoint.pth.tar",
        type=str)
    parser.add_argument(
        '--road_side_input_video_path',
        type=str,
        required=True)
    parser.add_argument(
        '--road_side_output_video_path',
        type=str,
        required=True)
    parser.add_argument(
        '--driver_input_video_path',
        type=str,
        required=True)
    parser.add_argument(
        '--driver_output_video_path',
        type=str,
        required=True)
    parser.add_argument("--evaluate_result", action="store_true")
    parser.add_argument(
        "--video_label_path",
        type=str)

    args = parser.parse_args()
    return args


def get_num(point_dict,name,axis):
    num = point_dict.get(f'{name}')[axis]
    num = float(num)
    return num

def cross_point(line1, line2):  
    x1 = line1[0]  
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1) 
    b1 = y1 * 1.0 - x1 * k1 * 1.0  
    if (x4 - x3) == 0: 
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def point_line(point,line):
    x1 = line[0]  
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1) 
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def point_point(point_1,point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return distance


def incabin_result(img, transform, plfd_backbone, m_yaw, direction, direction_momentum):
    height, width = img.shape[:2]
    model_test = AntiSpoofPredict(0)
    image_bbox = model_test.get_bbox(img)
    x1 = image_bbox[0]
    y1 = image_bbox[1]
    x2 = image_bbox[0] + image_bbox[2]
    y2 = image_bbox[1] + image_bbox[3]
    w = x2 - x1
    h = y2 - y1

    size = int(max([w, h]))
    cx = x1 + w/2
    cy = y1 + h/2
    x1 = cx - size/2
    x2 = x1 + size
    y1 = cy - size/2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    cropped = img[int(y1):int(y2), int(x1):int(x2)]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
    if cropped.shape[0] == 1:
        return 0, 0, 0, direction, direction_momentum, m_yaw


    cropped = cv2.resize(cropped, (112, 112))

    input = cv2.resize(cropped, (112, 112))
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = transform(input).unsqueeze(0).to(device)
    _, landmarks = plfd_backbone(input)
    pre_landmark = landmarks[0]
    pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
    point_dict = {}
    i = 0
    for (x,y) in pre_landmark.astype(np.float32):
        point_dict[f'{i}'] = [x,y]
        i += 1

    #yaw
    point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
    point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
    point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
    crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
    yaw_mean = point_point(point1, point31) / 2
    yaw_right = point_point(point1, crossover51)
    yaw = (yaw_mean - yaw_right) / yaw_mean
    yaw = int(yaw * 71.58 + 0.7037)

    #pitch
    pitch_dis = point_point(point51, crossover51)
    if point51[1] < crossover51[1]:
        pitch_dis = -pitch_dis
    pitch = int(1.497 * pitch_dis + 18.97)

    #roll
    roll_tan = abs(get_num(point_dict,60,1) - get_num(point_dict,72,1)) / abs(get_num(point_dict,60,0) - get_num(point_dict,72,0))
    roll = math.atan(roll_tan)
    roll = math.degrees(roll)
    if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
        roll = -roll
    roll = int(roll)

    #direction
    direction = "forward"
    if yaw > 0:
        direction = "right"
    elif yaw < -30:
        direction = "left"

    direction_momentum = "forward"
    momentum = 0.9
    m_yaw = yaw * momentum + m_yaw * (1-momentum)
    if m_yaw > 0:
        direction_momentum = "right"
    elif m_yaw < -30:
        direction_momentum = "left"

    return yaw, pitch, roll, direction, direction_momentum, m_yaw


def process_driver_view_input(args, driver_state):

    checkpoint = torch.load(args.driver_model_path, map_location=device)
    plfd_backbone = PFLDInference().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    videoCapture = cv2.VideoCapture(args.driver_input_video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps:",fps,"size:",size)
    videoWriter = cv2.VideoWriter(args.driver_output_video_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,size)

    success,img = videoCapture.read()
    # cv2.imwrite("1.jpg",img)
    m_yaw = -10
    direction = "forward"
    direction_momentum = "forward"
    while success:
        yaw, pitch, roll, direction, direction_momentum, m_yaw = incabin_result(img, transform, plfd_backbone, m_yaw, direction, direction_momentum)
        
        curr_state = State()
        curr_state.eye_direction = direction
        driver_state.append_state(curr_state)

        # cv2.putText(img,f"Head_Yaw(degree): {yaw}",(30,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        # cv2.putText(img,f"Head_Pitch(degree): {pitch}",(30,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        # cv2.putText(img,f"Head_Roll(degree): {roll}",(30,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        # cv2.putText(img,f"Looking {direction}",(30,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        # if direction_momentum == direction:
        #     cv2.putText(img,f"Looking (stable) {direction_momentum}",(30,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        # else:
        #     cv2.putText(img,f"Looking (stable) {direction_momentum}",(30,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)

        # videoWriter.write(img)
        success, img = videoCapture.read()
    
    print("#DriverView: There are {} states generated.".format(len(driver_state.states)))

def process_road_view_input(args, driver_state):
    objects_to_detect = ["person", "car", "truck", "stop sign"]
    objects_to_detect = ["stop sign"]

    detector = Detector(objects_to_detect)

    # predictions = detector.onImage(img_path)
    # pred_classes, pred_boxes = detector.filtered_outputs(img_path)

    # pred_classes, pred_boxes = detector.filtered_outputs(img_path)
    #print(pred_classes)
    #print(pred_boxes)

    road_side_output_video_name = args.road_side_output_video_path.split("/")[-1]
    road_side_output_video_prefix = args.road_side_output_video_path[:len(args.road_side_output_video_path) - len(road_side_output_video_name)]

    detector.onVideoNoOutput(args.road_side_input_video_path, road_side_output_video_name, road_side_output_video_prefix, driver_state)


    # vid_path = "/home/maxtom/ivan/project/Data/2021-10-17_Dashcam_Toguard_Mike/1_road/"
    # vid_name = "road_1.mp4"
    # out_path = "/home/maxtom/ivan/project/Data/2021-10-17_Dashcam_Toguard_Mike/output/"
    # detector.onVideo(vid_path, vid_name, out_path)

def process_speed(args, driver_state):
    print("process speed")


def calcuate_accuracy(labeled_state, driver_state):
    print("calcuate_accuracy")
    print(len(labeled_state.states))
    print(len(driver_state.states))

    eq = 0
    for i in range(len(labeled_state.states)):
        if labeled_state.states[i].eye_direction == driver_state.states[i].eye_direction and labeled_state.states[i].has_stop_sign == driver_state.states[i].has_stop_sign:
            eq = eq + 1
    return eq * 100.0 / len(labeled_state.states)

def main(args):
    args = parse_args()
    driver_state = DriverState()

    process_driver_view_input(args, driver_state)
    process_road_view_input(args, driver_state)
    process_speed(args, driver_state)

    print("============ Final driver states ===============")
    for s in driver_state.states:
        print(s)
    print("============ Final driver states ===============")

    strategy = StopSignEvaluationStrategy()
    res = strategy.evaluate(driver_state)

    if args.evaluate_result:
        print("============ Evaluting results ============")

        label_reader = LabelReader(5, 5)
        labeled_driver_state = label_reader.generate_driver_state(args.video_label_path)
        acc = calcuate_accuracy(driver_state, labeled_driver_state)
        print("====== accuracy is {}".format(acc))

    print("Is safe: ", res.final_label)
    print("has stop sign: ", res.has_stop_sign)
    print("looked left: ", res.looked_left)
    print("looked right: ", res.looked_right)


if __name__ == "__main__":
    args = parse_args()
    print("========= Args ==========")
    print(args)
    print("=========================")
    main(args)

