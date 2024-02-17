import pytest
from application import app as ai_app, db
from application.models import Prediction, User
from datetime import datetime
import json

'''Non-Api: 
    This are the test done without the use of api
'''
# 1. This test is to provide unexpected database connection error to test app against database failures
@pytest.mark.xfail(reason="Database connection error", strict=True)
@pytest.mark.usefixtures('authenticated_client', 'app_context')
def test_prediction_database(monkeypatch):
    def mock_commit(*args, **kwargs):
        raise Exception("Database connection error")
    
    monkeypatch.setattr(db.session, "commit", mock_commit)

    with pytest.raises(Exception) as exc_info:
        new_prediction = Prediction(userid=1, filename='image.png', model='A', predicted_on=datetime.now(), prediction='Tomato', confidence_score=0.98)
        db.session.add(new_prediction)
        db.session.commit()

    assert "Database error" in str(exc_info.value), "Unexpected failure"

# 2. This is to test for database failure here
@pytest.mark.xfail(reason="Query failure error", strict=True)
@pytest.mark.usefixtures('authenticated_client', 'app_context')
def test_query_failure(monkeypatch):
    def mock_filter_by(*args, **kwargs):
        raise Exception("Query operation failed")
    
    monkeypatch.setattr(db.session.query(Prediction).filter_by, "all", mock_filter_by)

    with pytest.raises(Exception) as exc_info:
        predictions = db.session.query(Prediction).filter_by(userid=1).all()

    assert "Query failed" in str(exc_info.value), "Querying database"