import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import requests
from tensorflow.keras.utils import Sequence, to_categorical



# Get the small image
test_data_small = tf.keras.utils.image_dataset_from_directory('../Dataset for CA1 part A/test'  ,
                                                   color_mode='rgb',
                                                   image_size=(31,31))
test_data_small


X_test_small = []
y_test_small = []

for images, labels in tqdm(test_data_small):
    images = tf.image.rgb_to_grayscale(images)
    X_test_small.append(images)
    y_test_small.append(labels)

X_test_small = np.concatenate(X_test_small, axis=0)
y_test_small = np.concatenate(y_test_small, axis=0)
print("Shape of the input data:", X_test_small.shape)
# Server URL
url = 'https://dl-model-o6iu.onrender.com/v1/models/small:predict'

def make_prediction(instances):
    data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    print(predictions)
    return predictions

def test_prediction():
    predictions = make_prediction(X_test_small[0:4])
    for i, pred in enumerate(predictions):
        assert y_test_small[i] == np.argmax(pred)

# Run the test
test_prediction()
