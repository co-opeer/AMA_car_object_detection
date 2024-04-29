import pandas as pd
from pathlib import Path

size_x = 380
size_y = 676
train_path = Path(r"C:\Users\PC\PycharmProjects\AMA_car_object_detection.git\dataset\data\training_images")
test_path = Path(r"C:\Users\PC\PycharmProjects\AMA_car_object_detection.git\dataset\data\testing_images")

train = pd.read_csv(
    r'C:\Users\PC\PycharmProjects\AMA_car_object_detection.git\dataset\data\train_solution_bounding_boxes.csv')
train[['xmin', 'ymin', 'xmax', 'ymax']] = train[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
train.drop_duplicates(subset='image', inplace=True, ignore_index=True)

