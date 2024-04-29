import cv2
import numpy as np
from keras_tuner.src.backend.io import tf
from matplotlib import pyplot as plt
import pandas as pd

from train_test.const import train_path, train, size_x, size_y


def data_generator(df=train, batch_size=20, path=train_path):
    while True:
        batch_indices = np.random.randint(0, df.shape[0], size=batch_size)
        images = np.zeros((batch_size, size_x, size_y, 3))
        bounding_box_coords = np.zeros((batch_size, 4))

        for i, idx in enumerate(batch_indices):
            row = df.loc[idx]
            image = cv2.imread(str(path / row.image)) / 255.
            images[i] = image
            bounding_box_coords[i] = np.array([row.xmin, row.ymin, row.xmax, row.ymax])

        yield {'image': images}, {'coords': bounding_box_coords}


def display_image(img, bbox_coords=[], pred_coords=[], norm=False):
    # if the image has been normalized, scale it up
    if norm:
        img *= 255.
        img = img.astype(np.uint8)

    # Draw the bounding boxes
    if len(bbox_coords) == 4:
        xmin, ymin, xmax, ymax = bbox_coords
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)

    if len(pred_coords) == 4:
        xmin, ymin, xmax, ymax = pred_coords
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])


def test_model(model, datagen):
    example, label = next(datagen)

    X = example['image']
    y = label['coords']

    pred_bbox = model.predict(X)[0]
    print(pred_bbox)

    img = X[0]
    gt_coords = y[0]
    print(gt_coords)

    display_image(img, pred_coords=pred_bbox, norm=True)


def test(model):
    datagen = data_generator(batch_size=1)

    for i in range(27):
        plt.figure(figsize=(15, 7))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            test_model(model, datagen)
        plt.show()


saved_model_path = r'\train_test\saved_model_good.h5'
loaded_model = tf.keras.models.load_model(saved_model_path)
test(loaded_model)
