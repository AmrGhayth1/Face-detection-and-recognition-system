# Face-detection-and-recognition-system

## 🛡️ Intelligent Face Privacy & Reporter Protection 
A computer vision project designed to automatically blur the faces of unrecognizable individuals (bystanders) in images and videos while preserving the visibility of specific "known" individuals (e.g., reporters or content creators). 

Example output: The system detects multiple faces, identifies the known subject, and blurs the unknown individuals to protect privacy.


## 📖 Project Overview 

This tool solves the problem of privacy in public filming. Using DeepFace for state-of-the-art facial recognition and OpenCV for image processing, the system scans media, identifies registered users (Reporters), and applies a dynamic blur filter to everyone else. 

**Key Features:**
- Reporter Registration: Easily add "safe" faces to a database using a folder of reference images. 
- Selective Blurring: Only blurs faces that do not match the registered "Reporter" embeddings. 
- Multi-Format Support: Works on both static Images and Videos. 
- High Accuracy: Uses DeepFace (SOTA) for embedding generation and cosine similarity for matching.
## 📂 File Structure 
This repository contains the main implementation script and several research notebooks documenting the development process. 
```
File "Name"                                                                   Description  
"implementation.py"                               The Main Application. A CLI tool to register reporters and process images/videos. 
"AVG_blur.py"                                     Helper module containing the custom Average Blur filter logic. 
"deepface.ipynb"                                  Research notebook experimenting with DeepFace embeddings and models. 
"face-detection-using-haar-cascade.ipynb"         Experiments with Haar Cascade classifiers for face detection. 
"recogination_opencv.ipynb"                       Tests using standard OpenCV recognition methods. 
"ZeroShot.ipynb"                                  Experiments with Zero-Shot learning techniques for recognition. 
```
## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/AmrGhayth1/Face-detection-and-recognition-system.git 
cd "your-reponame"  
```
2. Install dependencies:
This project requires Python 3.x and the following libraries: 
```bash 
pip install deepface opencv-python numpy pandas pillow
```
3. Directory Setup:

Ensure you have a folder named reporters in the root directory (the script will create it if it doesn't exist).

## 🚀 Usage 
### Run the main script using Python:
```bash
python implementation.py
```
The system will prompt you to select a mode:
1. Mode 1
```txt
Mode 1: Add New Reporter (Registration) 
Use this to "whitelist" a face so it doesn't get blurred.

1.Type 1 when prompted.
2.Enter the name of the reporter (e.g., Alice). 
3.Enter the path to a folder containing photos of that person.
4.The script will generate embeddings and save a .pkl file in the reporters/ directory.
```
2. Mode 2
```txt
Mode 2: Process Image 
1.Type 2 when prompted. 
2.Enter the path to the input image (e.g., test_cases/image1.jpg). 
3.The script will detect all faces, compare them against registered reporters, and blur unknown faces.
4.Output: Saved as output_blurred.jpg. 
Mode 3: Process Video 
1.Type 3 when prompted. 
2.Enter the path to the input video (e.g., test_cases/video1.mp4).
3.The script processes the video frame-by-frame.
4.Output: Saved as output_blurred_video.mp4.
```
### 🧪 Test Cases 
The project has been validated with the following scenarios:

1. Test Case 1: A group photo containing 0 Reporters + 3 strangers.


![test](https://github.com/user-attachments/assets/51946bf9-9116-49ad-a6e6-0dc7c5067ff1)



Output image 


![output_blurred](https://github.com/user-attachments/assets/a2b1e0a1-5208-4586-b25b-3356200626e9)



Output image after make operation "1" add new reporter (Alexandra Daddario) and input the same input 



![WhatsApp Image 2026-02-24 at 11 09 16 PM](https://github.com/user-attachments/assets/189ba5e8-d188-4dd0-a5d9-1e2e89e48a81)





2.Test Case 2

A moving shot of the Reporter walking through a crowd. (input vedio , operation 3) 



https://github.com/user-attachments/assets/3b988803-d36d-429c-954b-597b54e6acae



Output vedio



https://github.com/user-attachments/assets/78404e69-101a-4dcf-bb50-2646bee7b32a


Result: Real-time tracking and blurring of surrounding people throughout the video duration.

