import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='seperate')
    parser.add_argument(
        '--input_path',
        default="./merged_2.mp4",
        type=str)
    parser.add_argument(
        '--output_path',
        type=str,
        default="/home/maxtom/ivan/project/Data/2021-11-13_Merged_Nexar_Sam/separate/")
    parser.add_argument(
        "--output_fps",
        type=float,
        default=15.0)
    parser.add_argument("--get_incabin", action="store_true")
    parser.add_argument("--get_roadside", action="store_true")
    parser.add_argument("--get_speed", action="store_true")
    args = parser.parse_args()
    return args
    
def main(args):
    video = cv2.VideoCapture(args.input_path)
    if (video.isOpened() == False): 
        print("Error reading video file")
        return
        
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)
    fps = video.get(cv2.CAP_PROP_FPS)
    output_fps = min(fps,args.output_fps)

    fullname, _ = args.input_path.split(".mp4")
    _, ind = fullname.split("_")

    if args.get_incabin: videoWriter1 = cv2.VideoWriter(args.output_path+'incabin_{}_{}PFS.mp4'.format(ind, output_fps), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (1280,720))
    if args.get_roadside: videoWriter2 = cv2.VideoWriter(args.output_path+'road_{}_{}PFS.mp4'.format(ind, output_fps), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (1280,720))
    if args.get_speed: videoWriter3 = cv2.VideoWriter(args.output_path+'speed_{}_{}PFS.mp4'.format(ind, output_fps), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (332,720))

    success,frame = video.read()
    frame_id = 0
    while success:
        if (frame_id+1) % (fps // output_fps) == 0:
            incabin = frame[:,:1280,:]
            roadside = frame[:,1280:2560,:]
            speed = frame[:,2560:,:]
            if args.get_incabin: videoWriter1.write(incabin)
            if args.get_roadside: videoWriter2.write(roadside)
            if args.get_speed: videoWriter3.write(speed)

        success,frame = video.read()
        frame_id = (frame_id+1) % fps

    video.release()
    if args.get_incabin: videoWriter1.release()
    if args.get_roadside: videoWriter2.release()
    if args.get_speed: videoWriter3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    print("input_path:  ",  args.input_path)
    print("output_path: ",  args.output_path)
    print("output_fps:  ",  args.output_fps)
    print("get_incabin:  ", args.get_incabin)
    print("get_roadside:  ", args.get_roadside)
    print("get_speed:  ", args.get_speed)

    main(args)