import pandas as pd
import os
import subprocess
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip

def is_video_available(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        yt = YouTube(url)
        return True
    except Exception as e:
        print(f"Skipping {video_id}: Video unavailable - {e}")
        return False

def download_video(video_id, start_time, end_time, output_dir):
    url = f"https://www.youtube.com/watch?v={video_id}"
    file_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
        
        if not os.path.exists(file_path):
            print(f"Downloading {video_id}...")
            stream.download(output_dir, filename=f"{video_id}.mp4")
        else:
            print(f"{video_id} already downloaded.")
    except Exception as e:
        print(f"Pytube failed for {video_id}: {e}. Trying yt-dlp...")
        
        command = [
            "yt-dlp", "-f", "best", "-o", file_path, url
        ]
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            print(f"yt-dlp failed for {video_id}: {process.stderr}")
            return

    # Trim the video
    try:
        trimmed_path = os.path.join(output_dir, f"{video_id}_trimmed.mp4")
        clip = VideoFileClip(file_path).subclip(start_time, end_time)
        clip.write_videofile(trimmed_path, codec="libx264")
        clip.close()
        print(f"Trimmed video saved as {trimmed_path}")
    except Exception as e:
        print(f"Error trimming {video_id}: {e}")

# Load and process CSV files
datasets = {
    "train": ("countix_train.csv", 20),
    "val": ("countix_val.csv", 2),
    "test": ("countix_test.csv", 2)
}

output_directory = "downloaded_videos"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for split, (csv_file, limit) in datasets.items():
    split_dir = os.path.join(output_directory, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    
    df = pd.read_csv(csv_file).head(limit)  # Limit the number of downloads per dataset
    for _, row in df.iterrows():
        if is_video_available(row['video_id']):
            download_video(row['video_id'], row['repetition_start'], row['repetition_end'], split_dir)