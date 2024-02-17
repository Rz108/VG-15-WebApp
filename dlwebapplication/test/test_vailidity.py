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

#1. This is to validate those entry that are place into the database
@pytest.mark.usefixtures('authenticated_client', 'app_context')
@pytest.mark.parametrize("data", [
    {'userid': 1, 'filename': 'image1.png', 'model': 'A', 'predicted_on': datetime.now(), 'prediction': 'Tomato', 'confidence_score': 0.98, 'confidence_cat': 'High'},
    {'userid': 2, 'filename': 'image2.png', 'model': 'B', 'predicted_on': datetime.now(), 'prediction': 'Potato', 'confidence_score': 0.88, 'confidence_cat': 'Medium'},
    {'userid': 3, 'filename': 'image3.png', 'model': 'C', 'predicted_on': datetime.now(), 'prediction': 'Carrot', 'confidence_score': 0.95, 'confidence_cat': 'High'},
    {'userid': 1, 'filename': 'image4.png', 'model': 'A', 'predicted_on': datetime.now(), 'prediction': 'Cucumber', 'confidence_score': 0.75, 'confidence_cat': 'Low'},
    {'userid': 2, 'filename': 'image5.png', 'model': 'B', 'predicted_on': datetime.now(), 'prediction': 'Pepper', 'confidence_score': 0.85, 'confidence_cat': 'Medium'},
])
def test_add_prediction_entry_directly(data):
    new_prediction = Prediction(**data)

    db.session.add(new_prediction)
    db.session.commit()

    add_row = db.session.query(Prediction).filter_by(filename=data['filename']).first()

    assert add_row is not None, "Prediction should be added to the database"
    assert add_row.prediction == data['prediction'], f"The prediction should be '{data['prediction']}'"
    assert add_row.filename ==  data['filename']
    assert add_row.userid ==  data['userid']
    assert add_row.model ==  data['model']
    assert add_row.confidence_score ==  data['confidence_score']
    assert add_row.confidence_cat ==  data['confidence_cat']

    db.session.delete(add_row)
    db.session.commit()


'''API Test'''
#2. This is to validate those entry that are place into the database with the use of api here
@pytest.mark.usefixtures('authenticated_client', 'app_context')
def test_add_prediction_entry(authenticated_client):
    with authenticated_client.session_transaction() as session:
        authenticated_user_id = int(session.get('_user_id'))
    print(authenticated_user_id)
    data = {
        'filename': 'vegetable_image.png',
        'model': 'big',
        'prediction': 'Tomato',
        'confidence_score': 0.95,
        'confidence_cat':'High'
    }


    response = authenticated_client.post(f'/api/add/{authenticated_user_id}', data=json.dumps(data), content_type="application/json")
    assert response.status_code == 201, "Expected 201 Created status code"
    response_data = json.loads(response.data.decode('utf-8'))
    assert 'id' in response_data, "Response data should contain an 'id' field"

#3. This is to validate getting the correct entries back
# @pytest.mark.usefixtures('authenticated_client', 'app_context')
# def test_api_get_entry(authenticated_client):
#     with authenticated_client.session_transaction() as session:
#         authenticated_user_id = int(session.get('_user_id'))

#     data = {
#         'filename': 'vegetable_image.png',
#         'model': 'big',
#         'prediction': 'Tomato',
#         'confidence_score': 0.95,
#         'confidence_cat':'High'
#     }


#     response = authenticated_client.post(f'/api/add/{authenticated_user_id}', data=json.dumps(data), content_type="application/json")
#     response_data1 =  json.loads(response.data.decode('utf-8'))
#     ids = response_data1['id']
#     response = authenticated_client.get(f'/api/get/{1}')

#     print(response.data.decode('utf-8'))
#     print(response.status_code)
    
#     assert response.status_code == 200, "Success"
    
#     response_data = json.loads(response.data.decode('utf-8'))
#     print(response_data)
#     assert 'userid' in response_data, "This should not have an error"
#     assert response_data['id'] == 1, "Entry was not found"


#4. This is to check on the link whether it is able to pass the test and the status code
@pytest.mark.parametrize(
    "links",
    [
        ("/useless", 404),
        ("/", 200),
        ("/login", 200),
        ("/register", 200),
        ("/predict", 401),  
        ("/api/predict", 405),  
        ("/api/login", 405),
        ("/api/register", 405),
    ],
)
def test_links(client, links):
    link, code = links
    response = client.get(link)
    assert response.status_code == code

# 5. This is to to see if the links after loggin can be get
@pytest.mark.usefixtures('authenticated_client')
@pytest.mark.parametrize(
    "links",
    [
        ("/useless", 404),
        ("/", 200),
        ("/login", 200),  
        ("/register", 200), 
        ("/predict", 200),  

    ],
)
def test_logged_in_links(authenticated_client, links):
    link, code = links
    response = authenticated_client.get(link)
    assert response.status_code == code

# 6. Test for prediction base on data provided
def encode_images(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def collect_img(base_path="test/image_prediction", num_images=1):
    all_images = []
    base_path = Path(base_path)  
    all_classes = [class_folder for class_folder in os.listdir(base_path) if (base_path / class_folder).is_dir()]
    selected_classes = random.sample(all_classes, k=min(len(all_classes), 3))  # Select 3 classes randomly
    
    for class_folder in selected_classes:
        folder_path = base_path / class_folder
        images = os.listdir(folder_path)
        selected_images = random.sample(images, k=min(len(images), num_images))
        for image in selected_images:
            all_images.append((class_folder, str(folder_path / image)))
    return all_images

test_data = collect_img()

## 6.1 Test for big too here
@pytest.mark.usefixtures("authenticated_client")
@pytest.mark.parametrize("class_name,image_path", test_data)
def test_predict_vegetable_type_big(authenticated_client, class_name, image_path):
    encoded_image = encode_images(image_path)

    time.sleep(3)
    data = {
        "image": encoded_image,
        "modelChoice": "big"
    }
    response = authenticated_client.post("/api/predict", json=data)  
    assert response.status_code == 200, "Failed to authenticate or other error"
    response_body = response.json  
    assert "prediction" in response_body, "Response body should include a prediction field"
    assert response_body["prediction"] == class_name, f"Expected {class_name}, got {response_body['prediction']}"

## 6.2 Test for small too here
@pytest.mark.usefixtures("authenticated_client")
@pytest.mark.parametrize("class_name,image_path", test_data)
def test_predict_vegetable_type_small(authenticated_client, class_name, image_path):
    encoded_image = encode_images(image_path)

    time.sleep(3)
    data = {
        "image": encoded_image,
        "modelChoice": "small"
    }
    response = authenticated_client.post("/api/predict", json=data)  
    assert response.status_code == 200, "Failed to authenticate or other error"
    response_body = response.json  
    assert "prediction" in response_body, "Response body should include a prediction field"
    assert response_body["prediction"] == class_name, f"Expected {class_name}, got {response_body['prediction']}"

# 7. This is to test whether adding users is possiuble
def unique_email(base_email):
    timestamp = int(time.time())  
    return f"{base_email.split('@')[0]}_{timestamp}@{base_email.split('@')[1]}"

@pytest.mark.parametrize(
    "username, base_email, password",
    [
        (f'username{int(time.time())}', "email1@gmail.com", "password"),
        (f'username1{int(time.time())}', "email2@gmail.com", "password"),
        # ('usernam1', "email1@gmail.com", "password"), 
        # ('username2', "email2@gmail.com", "password"),  
    ],
)
def test_add_user(client, username, base_email, password, app_context):
    """Test adding a user with both dynamic and static data."""
    email = unique_email(base_email)  
    data = {'username': username, "email": email, "password": password, "password1": password}

    response = client.post("/api/register", json=data)
    assert response.status_code == 201, f"Expected status code 201 but got {response.status_code}"
    assert response.headers["Content-Type"] == "application/json", "Content-Type should be application/json"

    response_body = response.json
    assert "id" in response_body, "User ID was not returned"
    assert response_body["email"] == email, "Returned email does not match up"
    assert "joined_at" in response_body, "Joined At timestamp was not returned"


# 8. This is to test the ability to login here
@pytest.mark.parametrize(
    "userlist",[["email1@gmail.com", "password"],["email2@gmail.com", "password"],],
)
def test_user_login_api(client, userlist, capsys):
    with capsys.disabled():
        with client:
            data = {
                "email": userlist[0],
                "password": userlist[1],
            }

            response = client.post(
                "/api/login", data=json.dumps(data), content_type="application/json"
            )
            response_body = json.loads(response.get_data(as_text=True))
            print(response_body)
            assert response.status_code == 200
            assert response.headers["Content-Type"] == "application/json"
            assert response_body["email"] == userlist[0]

            with client.session_transaction() as sess:
                assert sess["user_id"] == response_body["id"]
