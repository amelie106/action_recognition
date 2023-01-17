import torch
import clip
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import pickle
import pandas as pd

# model
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# load data
df = pd.read_csv('data/csv/images_single.csv') # single image
# df = pd.read_csv('data/csv/images.csv') # 10 random frames

# feature extraction
total_image_features=[]
total_labels = []

for i, row in df.iterrows():
    print(i)
    # features
    image = preprocess(Image.open(row.image)).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    image_features=image_features.squeeze().detach().cpu().numpy()
    total_image_features.append(image_features.reshape((1,len(image_features))))

    # label
    # total_labels.append((row.caption, row.cleanliness, row.subject, row.trial))
    total_labels.append((row.caption, row.cleanliness, row.subject))

output_features=np.concatenate(total_image_features,axis=0)

print(np.shape(output_features))
print(len(total_labels))

with open('extracted/features_clip.pkl', 'wb') as f:
    pickle.dump(output_features, f)

with open('extracted/labels_clip.pkl', 'wb') as f:
    pickle.dump(np.array(total_labels), f)