from application import db
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import validates

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    userid = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  
    filename = db.Column(db.String, nullable=False)
    model = db.Column(db.String, nullable=False)
    predicted_on = db.Column(db.DateTime, nullable=False) 
    prediction = db.Column(db.String, nullable=False)
    confidence_score = db.Column(db.Float, nullable=True) 
    confidence_cat = db.Column(db.String, nullable=True) 

    @validates('filename', 'model', 'prediction', 'confidence_cat')
    def validate_strings(self, key, value):
        assert value != '', f'{key} cannot be empty'
        return value

    @validates('predicted_on')
    def validate_predicted_on(self, key, value):
        assert isinstance(value, datetime), f'{key} must be a datetime object'
        return value

    @validates('confidence_score')
    def validate_confidence(self, key, value):
        if value is not None:  
            assert 0 <= value <= 1, f'{key} must be between 0 and 1'
        return value

    @validates('userid')
    def validate_userid(self, key, value):
        assert value > 0, f'{key} must be greater than 0'
        return value

class User(UserMixin, db.Model):
    __tablename__ = "user"  
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), index=True, unique=True)
    email = db.Column(db.String(150), unique=True, index=True)
    password_hash = db.Column(db.String(150))
    joined_at = db.Column(db.DateTime(), default=datetime.utcnow, index=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)