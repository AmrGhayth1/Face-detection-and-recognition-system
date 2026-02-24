# implementation.py
import os
import cv2 as cv
import numpy as np
from PIL import Image
# from averageblur import blur_unknown_faces
# from facedetection import detect_faces, detect_faces_frame
import numpy as np
from pathlib import Path
from deepface import DeepFace
from typing import Tuple
import pandas as pd
import pickle 
from AVG_blur import AVG_filter
print("implementation.py started")

REPORTER_DIRECTORY = Path("reporters")
REPORTER_DIRECTORY.mkdir(exist_ok=True, parents=True)

BLUR_THRESHOLD = 0.5

def get_embeddings_from_folder(folder_path: Path) -> Tuple[np.array,np.array]:
    """
    Iterate throw all images in folder path
    and get all faces embeds foreach image
    @return embeddings (np.array)(num_faces, embedding_dim)
    """
    # img -> conv -> conv -> conv -> softmax -> [1 2 3 4] (num_imgs, embedding_dim)
    folder_image_paths = [
        p for p in folder_path.rglob("*") 
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    print(f"FOUND: {len(folder_image_paths)} images")
    folder_results =  [DeepFace.represent(path, enforce_detection=False) for path in folder_image_paths]
    embeddings = np.array([result['embedding']  for image_results in folder_results for result in image_results])

    if embeddings.ndim == 1:
       embeddings = embeddings.reshape(1, -1)


    return embeddings

def load_reporter_data():
    all_embeddings = []
    all_names = []
    for path in REPORTER_DIRECTORY.glob("*.pkl"):
        with open(path, "rb") as f:
            embeds = pickle.load(f) # [num_imgs_person, embed_dim]
            all_names.extend([path.stem]* embeds.shape[0])
            all_embeddings.append(embeds)
    all_embeddings = np.concatenate(all_embeddings)
    return all_embeddings, all_names
# -----------------------------
# Select mode
# -----------------------------
selectmode = input(
    "Type 1 to add new reporter, type 2 to process an image, type 3 to process a video: "
).strip()

# -----------------------------
# Reporter database paths
# -----------------------------
# -----------------------------
# 1: Add new reporter
# -----------------------------
if selectmode == "1":
    name = input("Enter reporter name: ").strip()
    folder_path = input("Enter folder path of reporter images: ").strip()
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    source_embeddings = get_embeddings_from_folder(folder_path) # num_images x embed_dim
    with open(REPORTER_DIRECTORY / f"{name}.pkl", "wb") as f:
        pickle.dump(source_embeddings, f)
# -----------------------------
# 2: Process image
# -----------------------------
elif selectmode == "2":
    image_path = input("Enter image path: ").strip()
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError("Image not found")
    # 1. get image embeds
    image = cv.imread(str(image_path))

    image_results = DeepFace.represent(str(image_path), enforce_detection=False)
    image_embeds = np.array([result['embedding'] for result in image_results]) # num_faces x embedding_dim
    # 2. get source/reporters embeds to hide
    reporters_embeds, reporters_names = load_reporter_data()
    print("REPORTERS SHAPE: ", reporters_embeds.shape, " IMAGE SHAPE: ", image_embeds.shape)
    # [num_reporters_images x embedding_dim] * [embedding_dim x num_faces_in_images]
    rep_norm = reporters_embeds / np.linalg.norm(reporters_embeds, axis=1, keepdims=True)
    img_norm = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
    similarity = img_norm @ rep_norm.T
    print("SIM MATRIX SHAPE: ", similarity.shape)
    # similarity[0] # first face image  similarity with all faces in reporter image

    blurred_image = image.copy()
    for i, face_result in enumerate(image_results):
        area = face_result['facial_area']
        x, y, w, h = area['x'], area['y'], area['w'], area['h']
        
        max_similarity_idx = similarity[i].argmax()
        max_similarity = similarity[i][max_similarity_idx]

        print(f"Best Similarity at {max_similarity} for {reporters_names[max_similarity_idx]}")
        if max_similarity <= BLUR_THRESHOLD:
           blurred_face = AVG_filter( blurred_image[y:y+h, x:x+w] , kernel_size=25)
           blurred_image[y:y+h, x:x+w] = blurred_face

    output_path = "output_blurred.jpg"
    cv.imwrite(output_path, blurred_image)
    print(f"📁 Output saved to {output_path}")

# -----------------------------
# 3: Process video
# -----------------------------
elif selectmode == "3":
    video_path = input("Enter video path: ").strip()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError("Video not found")

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError("Cannot open video")

    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    output_path = "output_blurred_video.mp4"
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load reporter embeddings once
    reporters_embeds, reporters_names = load_reporter_data()
    rep_norm = reporters_embeds / np.linalg.norm(reporters_embeds, axis=1, keepdims=True)

    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect faces and get embeddings
        frame_results = DeepFace.represent(frame, enforce_detection=False)
        if not frame_results:
            out.write(frame)
            continue

        frame_embeds = np.array([result['embedding'] for result in frame_results])
        img_norm = frame_embeds / np.linalg.norm(frame_embeds, axis=1, keepdims=True)
        similarity = img_norm @ rep_norm.T

        # 2. Blur unknown faces
        blurred_frame = frame.copy()
        for i, face_result in enumerate(frame_results):
            area = face_result['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']

            max_similarity_idx = similarity[i].argmax()
            max_similarity = similarity[i][max_similarity_idx]

            if max_similarity <= BLUR_THRESHOLD:
                blurred_face = AVG_filter(blurred_frame[y:y+h, x:x+w], kernel_size=25)
                blurred_frame[y:y+h, x:x+w] = blurred_face

        out.write(blurred_frame)

    cap.release()
    out.release()
    print(f"📁 Output saved to {output_path}")