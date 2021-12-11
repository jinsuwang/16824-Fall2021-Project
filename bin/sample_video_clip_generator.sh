# split source video into multiple 
python preprocess.py --input_path resource/input/merged_1.mp4 --output_path resource/output/ --output_fps 5 --get_incabin --get_roadside --get_speed 


# generate 10 second video clips.
ffmpeg -i "road_1_5.0PFS.mp4" -ss 00:00:0.0 -t 10 -an "road_1_5.0PFS_1s.mp4"
ffmpeg -i "incabin_1_5.0PFS.mp4" -ss 00:00:0.0 -t 10 -an "incabin_1_5.0PFS_1s.mp4"