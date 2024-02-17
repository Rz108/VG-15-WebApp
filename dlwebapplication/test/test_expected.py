import pytest
from application import app as ai_app, db
from application.models import Prediction, User
from datetime import datetime
import json
import time

'''Non-Api Test:
This is to test for api without the use of restapi
'''
@pytest.mark.usefixtures('authenticated_client','app_context')
@pytest.mark.xfail(reason="Duplicate email", strict=True)
# 1. This is to test for data that are duplicate
def test_email_exist():
    username = f'uniqueUser{int(time.time())}'  
    email = "testuser@gmail.com"

    user1 = User(username=username, email=email, joined_at=datetime.utcnow())
    user1.set_password('123')
    db.session.add(user1)
    db.session.commit()

    user2 = User(username=username, email=email, joined_at=datetime.utcnow())
    user2.set_password('123')

    db.session.add(user2)
    db.session.commit()

'''Api Test:
This is to test for api with the use of restapi
'''

# 1. This is to test for expected due to missing models in data
@pytest.mark.usefixtures('authenticated_client','app_context')
@pytest.mark.parametrize("data", [
    {'filename': '', 'model': 'big', 'prediction': 'Tomato', 'confidence_score': 0.95},  
    {'filename': 'vegetable_image.png', 'model': '', 'prediction': 'Tomato', 'confidence_score': 0.95},  
])

@pytest.mark.xfail(reason="Missing Models", strict=True)
def test_missing_models(authenticated_client, data, app_context):
    response = authenticated_client.post('/api/predict', data=json.dumps(data), content_type="application/json")
    assert response.status_code == 200, 'Models cannot be found here'

# 2. This is to test for expected due to missing values in data
@pytest.mark.usefixtures("authenticated_client",'app_context')
@pytest.mark.xfail(reason="Missing Values", strict=True)
def test_missing_values(authenticated_client, app_context):
    data = {'filename': 'vegetable_image.png', 'model': 'big'} 
    response = authenticated_client.post('/api/predict', data=json.dumps(data), content_type="application/json")
    assert response.status_code == 200, 'Values cannot be found here'

# 3. This is to test for expected due to id not exist in table
@pytest.mark.xfail(reason="Missing id in the prediction table", strict=True)
@pytest.mark.usefixtures('authenticated_client', 'app_context')
def test_get_entry_404(authenticated_client):

    non_id = db.session.query(db.func.max(Prediction.id)).scalar() + 1

    response = authenticated_client.get(f'/api/get/{non_id}')
    
    print(response.data.decode('utf-8'))
    print(response.status_code)

    assert response.status_code == 404, "Expected 404 Not Found status code"

    response_data = json.loads(response.data.decode('utf-8'))
    assert 'error' in response_data, "Response data is accurate"
    assert response_data['error'] == 'Entry not found', "Entry was found"


# 4. This is to test for duplicate data here
@pytest.mark.usefixtures("add_base_users")
@pytest.mark.xfail(reason="Expected failure due to duplicate email", strict=True)
@pytest.mark.parametrize(
    "username, email, password",
    [
        ('James4', "email1@gmail.com", "password"),
        ('James5', "email2@gmail.com", "password"),
    ],
)

def test_user_email_exist(client, username, email, password, app_context):
    data = {'username': username, "email": email, "password": password, "password1": password}
    response = client.post("/api/register", json=data)
    assert response.status_code == 409, "Expected a 409 Conflict due to duplicate email"

# 5. This is to test for invalid credentials
@pytest.mark.xfail(reason="Invalid login credentials")
@pytest.mark.parametrize(
    "userlist",
    [
        ["test@example.com", "jasss"],
        ["test1@example.com", "jasss"],
    ],
)
def test_user_fail_login_api(client, userlist, capsys):
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
