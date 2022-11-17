import numpy as np
import cv2

n_class = 100
classes = ['golf club', 'tent', 'rhinoceros', 'river', 'pliers', 'octagon', 'mug', 'cactus', 'saw', 'pig', 'basket', 'ear', 'stove', 'dragon', 'pineapple', 'peanut', 'drill', 'eraser', 'house plant', 'saxophone', 'donut', 'broccoli', 'snowflake', 'hamburger', 'sea turtle', 'line', 'crown', 'shovel', 'ice cream', 'mouse', 'power outlet', 'helmet', 'camera', 'knife', 'door', 'light bulb', 'mushroom', 'duck', 'table', 'broom', 'van', 'microwave', 'see saw', 'carrot', 'church', 'bicycle', 'mermaid', 'cloud', 'sleeping bag', 'ambulance', 'fan', 'rabbit', 'pear', 'couch', 'paint can', 'shark', 'swing set', 'blueberry', 'swan', 'potato', 'piano', 'The Great Wall of China', 'kangaroo', 'floor lamp', 'barn', 'cow', 'truck', 'traffic light', 'roller coaster', 'chandelier', 'angel', 'grapes', 'flip flops', 'speedboat', 'star', 'pillow', 'hedgehog', 'grass', 'zebra', 'asparagus', 'snowman', 'shoe', 'string bean', 'windmill', 'jacket', 'onion', 'submarine', 'fire hydrant', 'diving board', 'headphones', 'fence', 'bracelet', 'pond', 'elbow', 'beach', 'cup', 'giraffe', 'sink', 'sweater', 'tree']

BASE_SIZE = 256
# img shape : (224, 224, 1)
def draw_cv2img(raw_strokes, size=256, lw=6, time_color=True):
    x = np.zeros((size, size, 1))
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                        (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
    x[:, :, 0] = img
    x = np.repeat(x, 3, axis=2)
    return x

### 획마다 가중치
# img shape : (224, 224, 3)
def draw_cv2_weight(raw_strokes, size=256, lw=6, augmentation = False):
    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        points_count = len(stroke[0]) - 1
        grad = 255//points_count
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), (255, 255 - min(t,10)*13, max(255 - grad*i, 20)), lw)
    if size != BASE_SIZE:
        img = cv2.resize(img, (size, size))
    return img

# [70, 3]
def pad_sequences(strokes, seq_lengths):
    seq_tensor = np.zeros((seq_lengths, 3))
    for idx, stroke in enumerate(strokes):
        if idx == seq_lengths:
            break
        seq_tensor[idx, :] = stroke
        
    return seq_tensor