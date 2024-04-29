import numpy as np
import pandas as pd
import cv2



def csv_to_np():
    # Завантаження даних з CSV-файлу
    data = pd.read_csv(
        r'/dataset/data/train_solution_bounding_boxes.csv')

    # Перегляд перших кількох рядків для перевірки
    print(data.head())

    # Ітерація через кожен рядок та завантаження зображення та його міток
    images = []
    labels = []

    for index, row in data.iterrows():
        import os

        image_path = os.path.join(
            r"/dataset/data/training_images", row['image'])


        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        # Завантаження зображення та робимо обрізку, щоб взяти лише область рамки
        image = cv2.imread(image_path)
        cropped_image = image[int(ymin):int(ymax), int(xmin):int(xmax)]

        # Ресайз зображення до потрібних розмірів (якщо потрібно)
        cropped_image = cv2.resize(cropped_image, (150, 150))

        # Додавання зображення та його міток до списків
        images.append(cropped_image)
        labels.append([xmin, ymin, xmax, ymax])

    # Перетворення списків у масиви NumPy
    images = np.array(images)
    labels = np.array(labels)
    return images, labels



