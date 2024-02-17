import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import requests
from tensorflow.keras.utils import Sequence, to_categorical


# Get the big image
test_data_big = tf.keras.utils.image_dataset_from_directory('../Dataset for CA1 part A/test'  ,
                                                   color_mode='rgb',
                                                   image_size=(128,128))
test_data_big


X_test_big = []
y_test_big = []

for images, labels in tqdm(test_data_big):
    images = tf.image.rgb_to_grayscale(images)
    X_test_big.append(images)
    y_test_big.append(labels)

X_test_big = np.concatenate(X_test_big, axis=0)
y_test_big = np.concatenate(y_test_big, axis=0)
print("Shape of the input data:", X_test_big.shape)
# Server URL
url = 'https://dl-model-o6iu.onrender.com/v1/models/big:predict'

def make_prediction(instances):
    data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions

def test_prediction():
    predictions = make_prediction(X_test_big[0:5])
    print(X_test_big[0].shape)
    print(np.argsort(predictions)[::-1][:5])
    for i, pred in enumerate(predictions):
        assert y_test_big[i] == np.argmax(pred)

# Run the test
test_prediction()
