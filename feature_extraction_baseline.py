import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import skimage.measure
import os
import pickle
import pandas as pd
from transformers import BeitFeatureExtractor, BeitForImageClassification

feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

# model
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# load data
df = pd.read_csv('data/csv/images_single.csv') # single image
# df = pd.read_csv('data/csv/images.csv') # 10 random frames

# feature extraction
total_image_features=[]
total_labels = []

for i, row in df.iterrows():
    print(i)
    # features
    image = Image.open(row.image)
    inputs = feature_extractor(images=image, return_tensors="pt")
    image_features=inputs['pixel_values'].squeeze().detach().cpu().numpy()

    # reduce dimensionality
    image_features = image_features.reshape(1, (image_features.shape[0]*image_features.shape[1]*image_features.shape[2]))
    image_features = skimage.measure.block_reduce(image_features, (1,250), np.max)
    total_image_features.append(image_features)

    # label
    total_labels.append((row.caption, row.cleanliness, row.subject))
    
output_features=np.concatenate(total_image_features,axis=0)

print(np.shape(output_features))
print(len(total_labels))

with open('extracted/features_baseline.pkl', 'wb') as f:
    pickle.dump(output_features, f)

with open('extracted/labels_baseline.pkl', 'wb') as f:
    pickle.dump(np.array(total_labels), f)
