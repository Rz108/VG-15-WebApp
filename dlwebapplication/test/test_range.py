import pytest
from application import app as ai_app, db
from application.models import Prediction, User
from datetime import datetime
import json
import time


'''Non-Api: 
    This are the test done with the use of api
'''

# 1. This is to test for data that are out of the confidence score range
@pytest.mark.xfail(reason="Data out of range here for confidence score", strict=True)
@pytest.mark.parametrize("out_range_cs", [
    {'userid': 1, 'filename': 'image.png', 'model': 'A', 'predicted_on': datetime.now(), 'prediction': 'Tomato', 'confidence_score': -0.1},
    {'userid': 2, 'filename': 'image.png', 'model': 'A', 'predicted_on': datetime.now(), 'prediction': 'Tomato', 'confidence_score': 1.1},
])
def test_prediction_confidence_r(out_range_cs):
    with pytest.raises(ValueError):
        new_prediction = Prediction(**out_range_cs)
        db.session.add(new_prediction)
        db.session.commit()

# 2. This is to test for data that are out of the user id range
@pytest.mark.xfail(reason="Data out of range here for confidence score", strict=True)
@pytest.mark.parametrize("out_range_userid", [
    {'userid': -1, 'filename': 'image.png', 'model': 'A', 'predicted_on': datetime.now(), 'prediction': 'Tomato', 'confidence_score': -0.1},
    {'userid': -100, 'filename': 'image.png', 'model': 'A', 'predicted_on': datetime.now(), 'prediction': 'Tomato', 'confidence_score': 1.1},
])
def test_prediction_userid(out_range_userid):
    with pytest.raises(ValueError):
        new_prediction = Prediction(**out_range_userid)
        db.session.add(new_prediction)
        db.session.commit()

