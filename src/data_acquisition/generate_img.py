import os
import random

import cv2
import numpy as np


def augument(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    for _ in range(random.randint(4, 20)):
        pt1 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
        pt2 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        thickness = random.randint(3, 9)
        cv2.line(img, pt1, pt2, color, thickness)

    h, w, _ = img.shape
    top_left = (random.randint(0, w // 2), random.randint(0, h // 2))
    bottom_right = (
        top_left[0] + random.randint(15, 50),
        top_left[1] + random.randint(15, 50),
    )
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)

    cv2.imwrite(output_path, img)
