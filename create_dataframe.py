import pandas as pd
import os

# assign directory
directory = 'data/'
urls = []
captions = []
cleanliness = []
subject = []

## 10 random frames in video
# trial = []
# for subdir, dirs, files in os.walk(directory):
#     if 'images' in subdir:
#         for filename in files:
#             f = os.path.join(subdir, filename)
#             if os.path.isfile(f):
#                 urls.append(f)
#                 captions.append(filename.split('_')[0])
#                 cleanliness.append(filename.split('_')[1])
#                 subject.append(filename.split('_')[2])
#                 trial.append(filename.split('_')[3].split('.')[0])

# df = pd.DataFrame({'image': urls, 'caption': captions, 'cleanliness': cleanliness, 'subject': subject, 'trial': trial})
# df.to_csv('data/csv/images.csv')

# single images
for subdir, dirs, files in os.walk(directory):
    if 'images_single' in subdir:
        for filename in files:
            f = os.path.join(subdir, filename)
            if os.path.isfile(f):
                urls.append(f)
                captions.append(filename.split('_')[0])
                cleanliness.append(filename.split('_')[1])
                subject.append(filename.split('_')[2].split('.')[0])

df = pd.DataFrame({'image': urls, 'caption': captions, 'cleanliness': cleanliness, 'subject': subject})
df.to_csv('data/csv/images_single.csv')