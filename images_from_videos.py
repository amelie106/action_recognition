import cv2
import random
import os

directory = 'data/'

# 10 random frames
for subdir, dirs, files in os.walk(directory):
    if 'videos' in subdir:
        for filename in files:
            f = os.path.join(subdir, filename)
            if os.path.isfile(f):
                vidcap = cv2.VideoCapture(f)
                # get total number of frames
                totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                for i in range(0,10):
                    randomFrameNumber=random.randint(0, totalFrames)
                    # set frame position
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
                    success, image = vidcap.read()
                    image = cv2.flip(image, 0)
                    if success:
                        file = subdir.replace('videos', 'images') + '/' + filename.replace('.MOV', '') + '_' + str(i) + '.jpg'
                        print(file)
                        cv2.imwrite(file, image)

# # single image
# for subdir, dirs, files in os.walk(directory):
#     if 'videos' in subdir:
#         for filename in files:
#             f = os.path.join(subdir, filename)
#             if os.path.isfile(f):
#                 vidcap = cv2.VideoCapture(f)
#                 totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
#                 vidcap.set(cv2.CAP_PROP_POS_FRAMES,totalFrames/2)
#                 success, image = vidcap.read()
#                 image = cv2.flip(image, 0)
#                 if success:
#                     file = subdir.replace('videos', 'images_single') + '/' + filename.replace('.MOV', '') + '.jpg'
#                     print(file)
#                     cv2.imwrite(file, image)