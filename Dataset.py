import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import cv2
import glob
from tqdm import tqdm
import random
import os
import math
from torch.utils.data import Dataset, DataLoader, ConcatDataset


"""Creates one sequence from each video"""
class miniDataset(Dataset):
    
    def __init__(self, df, path_to_video):
        """
        df: DataFrame containing metadata for one video
        path_to_video: Path to the corresponding video file
        """
        self.path = path_to_video
        self.df = df.reset_index()
        self.count = self.df.loc[0, 'count']

    def getFrames(self, path=None):
        """Extract frames from a video file."""
        frames = []
        if path is None:
            path = self.path
        
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame from BGR to RGB for PIL compatibility
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {path}. Check the video file!")
        
        return frames

    def __getitem__(self, index):
        """Generates a sample with augmentation."""
        curFrames = self.getFrames()
        
        output_len = min(len(curFrames), random.randint(44, 64))
                
        newFrames = [curFrames[i * len(curFrames) // output_len - 1] for i in range(1, output_len + 1)]

        a = random.randint(0, 64 - output_len)
        b = 64 - output_len - a

        # Handle case where no synthetic data is used
        finalFrames = [newFrames[0] for _ in range(a)]
        finalFrames.extend(newFrames)        
        finalFrames.extend([newFrames[-1] for _ in range(b)])

        preprocess = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        Xlist = [preprocess(img).unsqueeze(0) for img in finalFrames]
        Xlist = [Xlist[i] if a < i < (64 - b) else torch.nn.functional.dropout(Xlist[i], 0.2) for i in range(64)]  
        X = torch.cat(Xlist)

        y = [0] * a
        y.extend([output_len / self.count if 1 < output_len / self.count < 32 else 0 for _ in range(output_len)])
        y.extend([0] * b)
        y = torch.FloatTensor(y).unsqueeze(-1)

        return X, y
        
    def __len__(self):
        return 1


class dataset_with_indices(Dataset):
    """Modifies Dataset class to return data, target, and index"""

    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        X, y = self.ds[index]
        return X, y, index

    def getPeriodDist(self):
        arr = np.zeros(32,)
        
        for i in tqdm(range(len(self))):
            _, p, _ = self[i]
            per = max(p)
            arr[per] += 1
        return arr

    def __len__(self):
        return len(self.ds)


def getCombinedDataset(dfPath, videoDir, split):
    """
    Loads the dataset and filters available videos.
    
    dfPath: Path to the CSV file.
    videoDir: Base directory where videos are stored.
    split: 'train' or 'test' to choose the correct subdirectory.
    """
    df = pd.read_csv(dfPath)

    video_folder = os.path.join(videoDir, split)  # Use correct subdirectory
    files_present = []

    for i in range(len(df)):
        video_id = df.loc[i, "video_id"]  # Fetch the correct filename from CSV
        path_to_video = os.path.join(video_folder, f"{video_id}.mp4")

        if os.path.exists(path_to_video):
            print(f"✅ Found: {path_to_video}")  # Debug print
            files_present.append(i)
        else:
            print(f"❌ Missing: {path_to_video}")  # Debug missing files

    if not files_present:
        raise FileNotFoundError(f"No valid video files found in {video_folder}. Check dataset paths!")

    # Filter dataframe to include only existing files
    df = df.iloc[files_present].reset_index(drop=True)

    miniDatasetList = []
    for i in range(len(df)):
        video_id = df.loc[i, "video_id"]
        path_to_video = os.path.join(video_folder, f"{video_id}.mp4")
        miniDatasetList.append(miniDataset(df.iloc[[i]], path_to_video))

    if not miniDatasetList:
        raise ValueError("miniDatasetList is empty! Ensure dataset paths are correct.")

    megaDataset = ConcatDataset(miniDatasetList)
    print(f"✅ Created dataset with {len(megaDataset)} videos.")
    
    return megaDataset 
