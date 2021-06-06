import os
import cv2
import time
import uuid

# define images to collect
labels = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
number_imgs = 5

# setup folders
IMAGE_PATH = os.path.join('tensorflow', 'workspace', 'images', 'collectionimages')
if not os.path.exists(IMAGE_PATH):
    if os.name == 'posix':
        !mkdir -p {IMAGE_PATH}
    if os.name == 'nt':
        !mkdir {IMAGE_PATH}

for label in labels:
    path = os.path.join(IMAGE_PATH, label)
    if not os.path.exists(path):
        !mkdir {path}

# capture images
for label in labels:
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(3)
    for img_idx in range(number_imgs):
        print('Collecting image {}'.format(img_idx))
        ret, frame = cap.read()
        img_name = os.path.join(IMAGE_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(img_name, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break()

cap.release()
cv2.destroyAllWindows()

# image labelling
LABEL_PATH = os.path.join('tensorflow', 'labelimg')
if not os.path.exists(LABEL_PATH):
    !mkdir {LABEL_PATH}
    !git clone https://github.com/tzutalin/labelImg {LABEL_PATH}

if os.name == 'posix':
    !make qt5py3
if os.name =='nt':
    !cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc

!cd {LABELIMG_PATH} && python labelImg.py

# move images for next step
TRAIN_PATH = os.path.join('tensorflow', 'workspace', 'images', 'train')
TEST_PATH = os.path.join('tensorflow', 'workspace', 'images', 'test')
ARCHIVE_PATH = os.path.join('tensorflow', 'workspace', 'images', 'archive.tar.gz')

!tar -czf {ARCHIVE_PATH} {TRAIN_PATH} {TEST_PATH}