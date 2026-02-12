import os
import numpy as np
import cv2
from keras_facenet import FaceNet
from mtcnn import MTCNN

# Initialize FaceNet and MTCNN
embedder = FaceNet()
detector = MTCNN()

def get_embedding(face_img):
    # Scale pixel values and add a dimension for the model
    face_img = face_img.astype('float32')
    # FaceNet expects (1, 160, 160, 3)
    face_img = cv2.resize(face_img, (160, 160))
    img_array = np.expand_dims(face_img, axis=0)
    
    # Predict the embedding
    embedding = embedder.embeddings(img_array)
    return embedding[0]

X, Y = [], []
dataset_path = 'dataset/'

print("Processing faces... this might take a minute.")

for student_name in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student_name)
    if not os.path.isdir(student_folder):
        continue
        
    for image_name in os.listdir(student_folder):
        img_path = os.path.join(student_folder, image_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # We assume the images are already cropped from collect_data.py
        # But we'll get embeddings for each
        embedding = get_embedding(img)
        X.append(embedding)
        Y.append(student_name)

# Save the processed data
np.savez_compressed('student_embeddings.npz', embeddings=np.array(X), names=np.array(Y))
print(f"Success! Saved embeddings for {len(np.unique(Y))} students.")