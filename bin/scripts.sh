
Look Left + Look right


Stop sign, car, pedestrian
Fully stop
positive
Only look left/right
Stop sign, car, pedestrian
Fully stop
Neutral
No look around
Stop sign, car, pedestrian
Fully stop 
Negative
All
Stop sign, car, pedestrian
Not fully stop
Negative



### Scenario - 1 Stop sign - happy path
1. 00:00 - 00:03 Entering into normal crossroad 
2. 00:03 - 00:04 look left
3. 00:04 - 00:05 look right
4. 00:04 - 00:06 turn right

### Scenario - 2 Stop sign - only look left 
1. 00:00 - 00:03 Entering into normal crossroad 
2. 00:03 - 00:04 look left
3. 00:04 - 00:06 turn right

### Scenario - 3 Stop sign - only look right  
1. 00:00 - 00:03 Entering into normal crossroad 
2. 00:03 - 00:04 look right
3. 00:04 - 00:06 turn right

### Scenario - 4 Stop sign - only look right  
1. 00:00 - 00:03 Entering into normal crossroad 
2. 00:03 - 00:04 turn right

### Scenario - 5 Driving Destraction - looking down
1. 00:00 - 00:03 - normal driving 
2. 00:03 - 00:05 - looking down

### Scenario - 6 Driving Destraction - looking down
1. 00:00 - 00:03 - normal driving 
2. 00:03 - 00:05 - looking down



ffmpeg -i Nexar-11-12-2021-07-48-55-A.MOV -i Nexar-11-12-2021-07-48-55-B.MOV -filter_complex hstack combined.mp4

ffmpeg \
  -i Nexar-11-12-2021-07-48-55-A.MOV \
  -i Nexar-11-12-2021-07-48-55-B.MOV \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map '[vid]' \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  combined_2.mp4



ffmpeg \
  -i Nexar-11-12-2021-08-05-01-A.MOV \
  -i Nexar-11-12-2021-08-05-01-B.MOV \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map '[vid]' \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  combined_2.mp4



ffmpeg \
  -i combined_2_trimmed.mp4 \
  -i speed_2_332_720_trimmed.mp4 \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map '[vid]' \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  combined_all_2.mp4




ffmpeg -i Nexar-11-12-2021-07-48-55-C.MP4 -vf scale=332:720 speed_332_720.mp4
ffmpeg -i Nexar-11-12-2021-08-05-01-C.MP4 -vf scale=332:720 speed_2_332_720.mp4


ffmpeg -i combined_all_1.mp4 -filter:v "crop=2892:720:0:0" -c:a copy merged_1.mp4
ffmpeg -i combined_all_2.mp4 -filter:v "crop=2892:720:0:0" -c:a copy merged_2.mp4

ffmpeg -i "incabin_1_5.0PFS.mp4" -ss 00:00:0.0 -t 10 -an "incabin_1_5.0PFS_10s.mp4"

ffmpeg -i "road_1_5.0PFS.mp4" -ss 00:00:0.0 -t 10 -an "road_1_5.0PFS_10s.mp4"


ffmpeg -i "road_1_5.0PFS.mp4" -ss 00:00:0.0 -t 1 -an "road_1_5.0PFS_1s.mp4"
ffmpeg -i "incabin_1_5.0PFS.mp4" -ss 00:00:0.0 -t 1 -an "incabin_1_5.0PFS_1s.mp4"


python preprocess.py --get_incabin --get_roadside --get_speed
python preprocess.py --input_path resource/input/merged_1.mp4 --output_path resource/output/ --output_fps 5 --get_incabin --get_roadside --get_speed 


KMP_DUPLICATE_LIB_OK=TRUE python video.py --video_name Nexar-11-12-2021-07-48-55-A.mp4

KMP_DUPLICATE_LIB_OK=TRUE python video.py --video_name outputincabin.mp4

KMP_DUPLICATE_LIB_OK=TRUE python video.py --video_name incabin.mp4

python video.py --video_name incabin.mp4 

python main.py \
  --road_side_input_video_path resource/input/road_1_5.0PFS_10s.mp4 \
  --driver_input_video_path resource/input/incabin_1_5.0PFS_10s.mp4 \
  --driver_output_video_path resource/output/processed_driver.mp4 \
  --road_side_output_video_path resource/output/processed_road_view.mp4


python main.py \
  --road_side_input_video_path resource/input/road_1_5.0PFS_1s.mp4 \
  --driver_input_video_path resource/input/incabin_1_5.0PFS_1s.mp4 \
  --driver_output_video_path resource/output/processed_driver.mp4 \
  --road_side_output_video_path resource/output/processed_road_view.mp4

conda create --name detectron2 --file requirements.txt


