# Repetitive Action Counting in YouTube Videos

## Dataset Description
I have used the **Countix dataset** for my assessment. The dataset consists of time-annotated YouTube videos, focusing on repetitive actions. Each row in the dataset represents a specific segment of a video where an action occurs multiple times. The dataset is stored in **CSV format** and contains the following columns:

- **video_id**: A unique identifier for each YouTube video.
- **kinetics_start**: The start time (in seconds) of the broader action segment in the original video.
- **kinetics_end**: The end time (in seconds) of the broader action segment in the original video.
- **repetition_start**: The start time (in seconds) of the repetitive action within the video.
- **repetition_end**: The end time (in seconds) of the repetitive action within the video.
- **count**: The number of times the action is repeated in the given segment.

### Dataset Processing
I used this dataset to extract specific segments from YouTube videos where repetitive actions occur. By utilizing the **repetition_start** and **repetition_end** timestamps, I was able to trim and save these relevant portions for further analysis.

### Directory Structure
The downloaded and trimmed videos are stored in a directory named `downloaded_videos/`, organized into three subsets:

- **Train Set (`countix_train.csv`)**: Used for model training (limited to **20 videos**).
- **Validation Set (`countix_val.csv`)**: Used for fine-tuning (limited to **2 videos**).
- **Test Set (`countix_test.csv`)**: Used for evaluation (limited to **2 videos**).

I used a limited amount of data due to computational constraints, as processing large numbers of videos requires significant storage, processing time, and system resources.

---

## Model Architecture: RepNet
**RepNet** is a deep learning model designed for **repetitive pattern detection** in videos. The architecture integrates **ResNet-50, transformer encoders, and multi-head attention mechanisms** to predict the periodicity and period length of video frames.

### Key Components
1. **Feature Extraction (ResNet-50 Backbone)**
   - A pre-trained **ResNet-50** extracts spatial features from input video frames.
   - The extracted features are taken from an intermediate layer of ResNet-50 (`layer3[2]`).

2. **3D Convolution for Temporal Representation**
   - A **3D convolutional layer** processes frame-wise feature maps, capturing temporal dependencies.
   - A **Batch Normalization (BN) layer** and **Max Pooling** are applied to refine representations.

3. **Self-Similarity Computation (Sims Module)**
   - Computes the **self-similarity matrix** of video frames, capturing periodic patterns.
   - Uses **Einsum operations** to calculate pairwise frame differences.
   - A **softmax operation** normalizes similarity scores.

4. **Multi-Head Attention on Similarities**
   - A **multi-head self-attention** layer further enhances frame similarity representations.
   - The output is concatenated with the computed **self-similarity matrix** for robust periodicity detection.

5. **Positional Encoding & Transformer Encoder Layers**
   - **Positional encoding** ensures temporal order is preserved.
   - Two **Transformer Encoder layers** process the temporal sequence to refine periodic patterns.

### Prediction Heads
- **Period Length Prediction**
  - A fully connected (**FC**) network estimates the repeating pattern's length in frames.
- **Periodicity Prediction**
  - Another **FC network** determines whether a periodic pattern is present in the input sequence.

### Forward Pass Workflow
1. Extract spatial features using **ResNet-50**.
2. Process features using **3D convolution** and pooling.
3. Compute frame similarities and apply **multi-head attention**.
4. Apply **Transformer Encoders** for temporal refinement.
5. Predict **period length** and **periodicity** using fully connected layers.

---

## Installation & Setup
### Clone the Repository
```bash
git clone git@github.com:fatemafaria142/Repetitive-Action-Counting.git
cd Repetitive-Action-Counting
```

## References
- [Counting Out Time: Class Agnostic Video Repetition Counting in the Wild](https://arxiv.org/pdf/2006.15418)




