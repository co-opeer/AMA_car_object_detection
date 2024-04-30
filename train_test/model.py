import cv2
import numpy as np

from keras_tuner.src.backend import keras
from tensorflow.keras import layers, models

from train_test.const import train, train_path, size_x, size_y

# Налаштування змінних для навчання моделі
batch_size = 16
epochs = 9
steps_per_epoch = 50

# Перевірка наявності збереженої моделі
saved_model_path = r'C:\Users\PC\PycharmProjects\AMA_car_object_detection\train_test\saved_model.h5'


def data_generator(df=train, path=train_path):
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


model = models.Sequential([
    layers.Input(shape=[size_x, size_y, 3], name='image'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='relu', name='coords')  # Вихідний шар для прогнозування координат рамок
])

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy']

)

history = model.fit(
    data_generator(),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs

)

print("Збереження навченої моделі...")
keras.saving.save_model(model, saved_model_path)
