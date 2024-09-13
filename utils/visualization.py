import numpy as np  
from data import *


def predict_visualization(im, boxes):
    mean = np.array(MEANS)

    im = im.cpu().detach().numpy().transpose(1,2,0)

    image = im + mean
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[0], image.shape[1]

    for i in range(boxes.shape[0]):
        xmin = int(boxes[i][0])
        ymin = int(boxes[i][1])
        xmax = int(boxes[i][2])
        ymax = int(boxes[i][3])
        
        if xmax > w or xmin < 0 or ymax > h or ymin < 0:
            continue

        image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 1)

    return image