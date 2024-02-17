import pytest
from application import app as ai_app, db
from application.models import Prediction, User
from datetime import datetime
import json
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from application.models import Prediction
from datetime import datetime
import pytest
import json
import pytest
import json
import base64
import os
import random
from pathlib import Path
import time

'''Non-Api: 
    This are the test done without the use of api
'''

# 1. This is to test for consistent such that the data added are consistent here
@pytest.mark.usefixtures('authenticated_client', 'app_context')
def test_add():
    data = {'userid': 1, 'filename': 'image.png', 'model': 'B', 'predicted_on': datetime.now(), 'prediction': 'Lettuce', 'confidence_score': 0.95}
    first = Prediction(**data)
    db.session.add(first)
    db.session.commit()

    second = Prediction(**data)
    db.session.add(second)
    db.session.commit()

    db.session.delete(first)
    db.session.commit()

    db.session.delete(second)
    db.session.commit()

    assert  first.id != second.id, 'Duplicate entries seen here'


'''Api: 
    This are the test done with the use of api
'''
# 2. This is to test for api with consistently able to predict same result
def encode_images(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def collect_img(base_path="test/image_prediction", num_images=1):
    all_images = []
    base_path = Path(base_path)  
    all_classes = [class_folder for class_folder in os.listdir(base_path) if (base_path / class_folder).is_dir()]
    selected_classes = random.sample(all_classes, k=min(len(all_classes), 3))  
    
    for class_folder in selected_classes:
        folder_path = base_path / class_folder
        images = os.listdir(folder_path)
        selected_images = random.sample(images, k=min(len(images), num_images))
        for image in selected_images:
            all_images.append((class_folder, str(folder_path / image)))
    return all_images

test_data = collect_img()
@pytest.mark.usefixtures("authenticated_client")
@pytest.mark.parametrize("class_name,image_path", test_data[:1])  
def test_prediction_consistency(authenticated_client, class_name, image_path):
    encoded_image = encode_images(image_path)
    data = {"image": encoded_image, "modelChoice": "big"}
    
    for _ in range(5):
        response = authenticated_client.post("/api/predict", json=data)  
        assert response.status_code == 200
        response_body = response.json  
        assert "prediction" in response_body
        assert response_body["prediction"] == class_name, "Prediction should be consistent across multiple times"