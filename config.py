

DATA_ROOT='./data'
IMAGE_SIZE=(448, 448)
BATCH_SIZE=32
MIN_CONFIDENCE=0.2
EPSILON = 1E-6
EPOCHS=200
LEARNING_RATE=1E-4
OBJ_INDEX={'bird': 2, 'tvmonitor': 19, 'sofa': 17, 'chair': 8, 'aeroplane': 0, 'person': 14, 'pottedplant': 15, 'car': 6, 'train': 18, 'motorbike': 13, 'cat': 7, 'cow': 9, 'horse': 12, 'bus': 5, 'dog': 11, 'diningtable': 10, 'bicycle': 1, 'bottle': 4, 'boat': 3, 'sheep': 16}
NUM_WORKERS=2