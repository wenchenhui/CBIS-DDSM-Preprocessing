import json
from collections import namedtuple

import cv2
import pandas as pd
import pydicom

try:
    import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return iter(it)

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])

BRECTS_AREA_TRESH = 10
IM_DESCRIPTIONS_PATH = 'descriptions.csv'
LABELS_PATH = 'labels.json'


def get_bounding_rects(mask_path):
    mask = pydicom.dcmread(mask_path).pixel_array
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = (Rect(*cv2.boundingRect(contour)) for contour in contours)
    return [rect for rect in rects if rect.w * rect.h > BRECTS_AREA_TRESH]


def scale_rect(rect, fx, fy):
    return Rect(rect.x * fx, rect.y * fx, rect.w * fx, rect.h * fy)


def show_bounding_rects(im_path):
    im = pydicom.dcmread(im_path).pixel_array
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    color = int(im.max())
    thickness = max(im.shape) // 200
    fontScale = 5
    for (x, y, w, h) in labels[im_path]['bounding_rects']:
        cv2.rectangle(im, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(im, labels[im_path]['pathology'], (x, y - 2 * thickness),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
    import matplotlib.pyplot as plt
    plt.imshow(im, cmap='gray')
    plt.show()

if __name__ == "__main__":
    im_descriptions = pd.read_csv(IM_DESCRIPTIONS_PATH)
    labels = {
        im.path: {
            'bounding_rects': get_bounding_rects(im.mask_path),
            'pathology': im.pathology,
        } for im in tqdm(im_descriptions.itertuples())
    }
    with open(LABELS_PATH, mode='w') as f:
        json.dump(labels, f)
