from application import app
from flask import render_template, request, flash, redirect, url_for,   jsonify, session
from application.forms import Login ,RegistrationForm , ResetPasswordForm, ChangePasswordForm
from application import manager
from application import db
from application.models import Prediction, User
from datetime import datetime
import pandas as pd
import numpy as np
from flask_login import login_user , logout_user , login_required , current_user, LoginManager
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import Forbidden
from sqlalchemy.exc import IntegrityError
from werkzeug.utils import secure_filename
import pytz
import json
import os
import time
from sqlalchemy import desc, asc
import base64
from io import BytesIO
import requests
from PIL import Image
from sqlalchemy import or_ , desc
import tensorflow as tf
import uuid
from collections import Counter
from sqlalchemy import func, Integer
from datetime import datetime, date
import hashlib
import binascii

# Set the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Set the login manager
login_manager = LoginManager()
login_manager.login_view = 'login'  
login_manager.init_app(app)

# Set the loader methods
@login_manager.user_loader
def loader(userid):
    return User.query.get(int(userid))

# set the home about us page here
@app.route('/')
def home():
    return render_template('home.html')


# Set the registration methods
@app.route('/register', methods=['GET', 'POST'])
def register():

    # Set up the form for registration
    form = RegistrationForm()
    # If it is a post method
    if request.method == 'POST':
        if form.validate_on_submit():

            # Check for existing user here

            existing_user = User.query.filter_by(email=form.email.data).first()
            existing_user_username = User.query.filter_by(username=form.username.data).first()
            # If there is no existing user
            if existing_user is None and existing_user_username is None:

                # It sets the password and 
                user = User(username=form.username.data, email=form.email.data)

                # Set the password after validation of password 
                user.set_password(form.password_1.data)

                # Commit to the database
                db.session.add(user)
                db.session.commit()

                # Redirect the user to login page here
                return redirect(url_for('login'))
            
            # IF the user already exist, warmog will be shown
            else:
                flash('A user with that email or username already exists.')
    
    # Return the render template here
    return render_template('register.html', form=form)


# def parse_scrypt_hash(hash_string):
#     algorithm_params, salt, hash_value = hash_string.split('$')
#     # Split the first part by ':' and use the first three values after 'scrypt'
#     # as n, r, p parameters
#     _, n, r, p = algorithm_params.split(':', 3)
#     # Convert n, r, p to integers
#     n, r, p = int(n), int(r), int(p)
#     return n, r, p, salt, hash_value

# def verify_scrypt_password(password, hash_string):
#     n, r, p, salt, stored_hash = parse_scrypt_hash(hash_string)
#     # Increase maxmem if necessary, here using 512MB as an example
#     maxmem = 512 * 1024 * 1024  # Adjust based on your requirements
#     password_hash = hashlib.scrypt(password.encode('utf-8'), salt=salt.encode('utf-8'), n=n, r=r, p=p, dklen=64, maxmem=maxmem)
#     password_hash_hex = binascii.hexlify(password_hash).decode('utf-8')
#     return password_hash_hex == stored_hash
# Set the login route
@app.route('/login', methods=['GET','POST'] )
def login():

    # Set up the login form here
    form = Login()

    # If methods is equal to post
    if request.method == 'POST':

        # Validate on submit here
        if form.validate_on_submit():

            # Filter the uyser by their id
            user = User.query.filter_by(email = form.email.data ).first()
            # Debugging tools
            # print(user)
            # Check whether the password is correct
            if user:
                # if verify_scrypt_password(form.password.data, user.password_hash):
                # Utilise password hash to hash the password so that the company is not able to see
                if check_password_hash(user.password_hash , form.password.data):
                    print('login')
                    login_user(user)

                    # Redirect to predict
                    return redirect('/predict')
                
                # If wrong password here
                else:
                    flash('Wrong password' , 'danger')
            # If user does not exist here
            else:
                flash('User does not exist' , 'danger')
        
        # Unexpected error
        else:
            flash('Error, Unable to login' , 'danger')
    
    # Return the rendered template here
    return render_template("login.html", form=form, title="Sign Up" , index = True )

# To save the image into a specific path
def save_inputImage(imageName, file):
    filename = f"{int(time.time())}_{imageName}"
    filepath = f'./application/static/stored_images/{filename}'
    file.save(filepath)
    return filename

# Setting the add entry methods here
def add_entry(new_entry):
    try:
        db.session.add(new_entry)
        db.session.commit()
        return new_entry.id
    except Exception as error:
        print(error)
        db.session.rollback()
        flash(error,"danger")

def make_prediction(instances,url):
    data = json.dumps({"signature_name": "serving_default", "instances":instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    print(json_response.text)
    print(json.loads(json_response.text))
    predictions = json.loads(json_response.text)['predictions']
    return predictions


classes = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli', 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}
@app.route("/predict", methods=['GET', 'POST'])
@login_required
def predict():

    if not current_user.is_authenticated:
        flash('You have not logged in yet.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST': 
        image = None
        fName = ''
        try:
            if 'file' in request.files and request.files['file'].filename:
                file = request.files['file']
                fName = secure_filename(file.filename)
                original_save_path = os.path.join('application/static/stored_images/', fName)
                file.save(original_save_path)
                image = tf.io.read_file(original_save_path)
                image = tf.image.decode_image(image, channels=1)  

            elif 'webcamImage' in request.form:
                baseImage = request.form['webcamImage']
                if baseImage:
                    image_data = base64.b64decode(baseImage.split(',')[1])
                    image = tf.image.decode_image(image_data, channels=1)  
                    fName = 'webcam_capture.png'
                    original_save_path = os.path.join('application/static/stored_images/', fName)
                    tf.io.write_file(original_save_path, tf.io.encode_png(image))

            model_choice = request.form.get('modelChoice')
            if image is None:
                flash('No image provided', 'red')
                return redirect(url_for('predict'))
            
            if model_choice == 'big':
                image = tf.image.resize(image, [128, 128])
                model_url = 'https://dl-model-o6iu.onrender.com/v1/models/big:predict'
            else:
                image = tf.image.resize(image, [31, 31])
                model_url = 'https://dl-model-o6iu.onrender.com/v1/models/small:predict'

            image = image.numpy()  
            image = np.expand_dims(image, axis=0) 
            predictions = make_prediction(image , model_url)
            probData = {
                expression: probability for expression, probability in zip(classes.values(), predictions[0])
            }
            print('here2')

            utc = datetime.utcnow()
            TimeSgt = pytz.timezone('Asia/Singapore')
            SGTTime = utc.replace(tzinfo=pytz.utc).astimezone(TimeSgt)
            for i, pred in enumerate(predictions):
                pred_best = np.argmax(pred)
            print('here2')
            predictions = make_prediction(image, model_url)
            print('here3')

            pred_best = np.argmax(predictions[0])
            confidence_score = np.max(predictions[0])

            confidence_itv = 'Low'
            if confidence_score*100 > 80:
                confidence_itv = 'High'
            elif confidence_score*100 > 50:
                confidence_itv = 'Medium'
            prediction = Prediction(
                userid = current_user.id,
                filename=fName,
                model='128x128 Model' if model_choice == 'big' else '31x31 Model',
                predicted_on=SGTTime,
                prediction=classes[pred_best],
                confidence_score=confidence_score,
                confidence_cat = confidence_itv  # Save the confidence score
            )

            pred_id = add_entry(prediction)
        except Exception as e:
            # Handle generic exceptions, which include errors in image processing
            flash(f'An error occurred while processing the image: {str(e)}', 'danger')
            return redirect(url_for('predict'))
        # Return the template with prediction and confidence score
        return render_template('index.html', title='Predict Vegetable Type', prediction=probData, confidence=confidence_score, classes=classes, best = classes[pred_best])

    # Return the render template here for GET requests
    return render_template("index.html", title='Predict Vegetable Type', prediction=None, classes=classes)


# Setting the remove entry methods here
def remove_entry(id):
    try:
        entry = Prediction.query.get_or_404(id)
        db.session.delete(entry)
        db.session.commit()
        flash("Entry removed successfully", "success")
    except Exception as error:
        db.session.rollback()
        flash(str(error), "danger")

# Setting the get entry here
def get_entry(id):
    try:
        result = db.get_or_404(Prediction, id)
        return result
    except Exception as error:
        db.session.rollback()
        flash(str(error), "danger")
        return 0



import re
from sqlalchemy import literal_column

@app.route('/history', methods=['GET', 'POST'])
@login_required
def history():
    # The check for current_user.is_authenticated is not necessary because @login_required already covers it
    if request.method == 'POST':
        id_to_delete = request.form.get('id_to_delete')
        if id_to_delete:
            prediction = Prediction.query.filter_by(id=id_to_delete, userid=current_user.id).first()
            if prediction:
                db.session.delete(prediction)
                db.session.commit()
                flash('Prediction deleted successfully.', 'success')
            else:
                flash('No prediction found to delete.', 'error')
            return redirect(url_for('history'))

    page = request.args.get('page', 1, type=int)
    sortBy = request.args.get('sort', 'id', type=str)
    sortOrder = request.args.get('order', 'desc', type=str)
    orderFunc = desc if sortOrder == 'desc' else asc

    query = Prediction.query.filter(Prediction.userid == current_user.id)
    
    # Filters here
    modelFilter = request.args.get('model_filter')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    confidence_min = request.args.get('confidence_min', type=float)
    confidence_max = request.args.get('confidence_max', type=float)
    vegetable_filter = request.args.get('vegetable_filter')
    confidence_filter = request.args.get('confidence_filter')

    searchQuery = request.args.get('q', type=str)
    if searchQuery:
        query = query.filter((Prediction.filename.like(f'%{searchQuery}%')) | 
                             (Prediction.prediction.like(f'%{searchQuery}%')))

    if modelFilter:
        query = query.filter(Prediction.model == modelFilter)
    if date_from:
        query = query.filter(Prediction.predicted_on >= datetime.strptime(date_from, '%Y-%m-%d'))
    if date_to:
        query = query.filter(Prediction.predicted_on <= datetime.strptime(date_to, '%Y-%m-%d'))
    if confidence_min is not None:
        query = query.filter(Prediction.confidence_score >= confidence_min/100)
    if confidence_max is not None:
        query = query.filter(Prediction.confidence_score <= confidence_max/100)
    if vegetable_filter:
        query = query.filter(Prediction.prediction == vegetable_filter)
    if confidence_filter:
        query = query.filter(Prediction.confidence_cat == confidence_filter)
    
    predictions = query.order_by(orderFunc(getattr(Prediction, sortBy)))\
                        .paginate(page=page, per_page=5, error_out=False)
    
    next_url = url_for('history', page=predictions.next_num, sort=sortBy, order=sortOrder) if predictions.has_next else None
    prev_url = url_for('history', page=predictions.prev_num, sort=sortBy, order=sortOrder) if predictions.has_prev else None

    return render_template('history.html', predictions=predictions, next_url=next_url, prev_url=prev_url, classes=classes)


# Set the visualisation link here
@app.route('/visualisation', methods=['GET'])
@login_required
def visualisation():
    if not current_user.is_authenticated:
        flash('You have not logged in yet.', 'warning')
        return redirect(url_for('login'))

    model_filter = request.args.get('model_filter', 'all')

    query = Prediction.query.filter_by(userid=current_user.id)

    # Filter base on the type of models
    if model_filter != 'all':
        query = query.filter_by(model=model_filter)

    class_distribution = query.with_entities(
        Prediction.prediction, func.count(Prediction.id)
    ).group_by(Prediction.prediction).all()

    predictions_over_time = query.with_entities(
        func.date(Prediction.predicted_on).label('date'), func.count(Prediction.id)
    ).group_by('date').all()

    confidence_distribution = query.with_entities(
        func.round(Prediction.confidence_score * 10) / 10, func.count(Prediction.id)
    ).group_by(func.round(Prediction.confidence_score * 10) / 10).all()

    if class_distribution:
        class_distribution_labels, class_distribution_values = zip(*class_distribution)
    else:
        class_distribution_labels, class_distribution_values = [], []

    if predictions_over_time:
        predictions_over_time_labels, predictions_over_time_values = zip(
            *[(d.strftime("%Y-%m-%d") if isinstance(d, (datetime, date)) else d, count)
              for d, count in predictions_over_time])
    else:
        predictions_over_time_labels, predictions_over_time_values = [], []

    confidence_distribution_data = {
        'labels': [str(confidence) for confidence, _ in confidence_distribution],
        'data': [count for _, count in confidence_distribution],
    }

    return render_template(
        'visualisation.html',
        model_filter=model_filter,  
        class_distribution_labels=class_distribution_labels,
        class_distribution_values=class_distribution_values,
        predictions_over_time_labels=predictions_over_time_labels,
        predictions_over_time_values=predictions_over_time_values,
        confidence_distribution_data=confidence_distribution_data,
    )


# Setting the logout options here
@app.route('/logout', methods=['GET'])
def logout():
    logout_user()
    return redirect('/login')


# Set the methods for reset password
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():

    # Start the reset password form
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if not user:
            flash('User not found!', 'danger')
            return redirect(url_for('login'))
        
        # Settubg the new password
        user.set_password(form.new_pw.data)
        db.session.commit()
        flash('Password has been reset!', 'success')
        return redirect(url_for('login'))

    # Return render template
    return render_template('reset_password.html', form=form)

# Set the methods to get the profile
@app.route('/profile')
def profile():
    userid = current_user.id  
    user = User.query.get_or_404(userid)
    return render_template('profile.html', user=user)


# Set the change password by username
@app.route('/change-password/<username>', methods=['GET', 'POST'])
def change_password(username):
    user = User.query.filter_by(username=username).first_or_404()
    form = ChangePasswordForm()
    if form.validate_on_submit():
        user.set_password(form.new_password.data)
        db.session.commit()
        flash('Your password has been updated!', 'success')
        return redirect(url_for('profile', username=username))
    
    # Return the rendered template
    return render_template('change_password.html', form=form, username=username)


## Restful api
# Prediction here
def make_prediction(instances, url):
    if hasattr(instances, "tolist"):
        instances = instances.tolist()
    data = json.dumps({"signature_name": "serving_default", "instances": instances})
    headers = {"content-type": "application/json"}
    response = requests.post(url, data=data, headers=headers)

    print(response.text)

    predictions = json.loads(response.text).get('predictions', [])

    return predictions

# Set the methods for predict
@app.route("/api/predict", methods=['POST'])
@login_required
def api_predict():
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json()
    if not data or 'image' not in data or not data['image']:
        return jsonify({"error": "No data provided"}), 400
    if 'image' not in data or 'modelChoice' not in data:
        return jsonify({"error": "Image or model choice not provided"}), 400

    try:
        baseImage = data['image']
        model_choice = data['modelChoice']
        image_data = base64.b64decode(baseImage)
        image = tf.image.decode_image(image_data, channels=1)  
    except Exception as e:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        if model_choice == 'big':
            image = tf.image.resize(image, [128, 128])
            model_url = 'https://dl-model-o6iu.onrender.com/v1/models/big:predict'
        else:
            image = tf.image.resize(image, [31, 31])
            model_url = 'https://dl-model-o6iu.onrender.com/v1/models/small:predict'

    except Exception as e:
        return jsonify({"error": "Error processing image"}), 500

    image = image.numpy()
    image = np.expand_dims(image, axis=0)
    predictions = make_prediction(image, model_url)

    try:
        pred_best = np.argmax(predictions[0])
        confidence_score = np.max(predictions[0])
        unique_id = str(uuid.uuid4())
        predicted_class_name = classes[pred_best].replace("_", "-")
        filename = f"{predicted_class_name}_{unique_id}.png"

        utc = datetime.utcnow()
        TimeSgt = pytz.timezone('Asia/Singapore')
        SGTTime = utc.replace(tzinfo=pytz.utc).astimezone(TimeSgt)

        confidence_itv = 'Low'
        if confidence_score*100 > 80:
            confidence_itv = 'High'
        elif confidence_score*100 > 50:
            confidence_itv = 'Medium'
        prediction_entry = Prediction(
            userid = current_user.id,
            filename=filename,
            model='128x128 Model' if model_choice == 'big' else '31x31 Model',
            predicted_on=SGTTime,
            prediction=classes[pred_best],
            confidence_score=confidence_score,
            confidence_cat = confidence_itv 
        )
        pred_id = add_entry(prediction_entry)
    except Exception as e:
        return jsonify({"error": "Error saving prediction to database"}), 500

    return jsonify({
        "prediction": classes[pred_best],
        "confidence": float(confidence_score),
        "filename": filename
    })



# Set the methods for adding
@app.route("/api/add/<int:user_id>", methods=['POST'])
@login_required
def api_add(user_id):
    try:

        data = request.get_json()

        filename = data['filename']
        model = data['model']
        prediction = data['prediction']
        confidence_score = data.get('confidence_score')  

        new_entry = Prediction(
            userid=user_id,
            filename=filename,
            model=model,
            prediction=prediction,
            predicted_on=datetime.utcnow(),
            confidence_score=confidence_score
        )

        db.session.add(new_entry)
        db.session.commit()
        
        return jsonify({'id': new_entry.id}), 201  #
    except KeyError as e:

        return jsonify({'error': f'Missing data: {e}'}), 400
    except Exception as e:

        return jsonify({'error': str(e)}), 500


def get_entry(id):
    try:
        # Version 2
        result = db.get_or_404(Prediction, id)
        return result
    except Exception as error:
        db.session.rollback()
        flash(str(error), "danger")
        return 0

@app.route("/api/get/<int:id>", methods=['GET'])
@login_required
def api_get(id):
    prediction = get_entry(id)
    if prediction is None:
        return jsonify({'error': 'Entry not found'}), 404
    print(prediction)

    data = {
        'id': prediction.id,
        'userid': prediction.userid,
        'filename': prediction.filename,
        'model': prediction.model,
        'prediction': prediction.prediction,
        'confidence_score': prediction.confidence_score,
        'predicted_on': prediction.predicted_on.strftime('%Y-%m-%d %H:%M:%S') 
    }
    return jsonify(data), 200


# Set the methods for registration
class CustomDatabaseError(Exception):
    def __init__(self, message="There was a database error"):
        self.message = message
        super().__init__(self.message)

class CustomApplicationError(Exception):
    def __init__(self, message="An unexpected error occurred in the application"):
        self.message = message
        super().__init__(self.message)

@app.route("/api/register", methods=['POST']) 
def api_register():
    try:
        data = request.get_json()
        username,email = data['username'], data['email']
        password, password1 = data['password'], data['password1']
        if password != password1:
            return jsonify({'Error':f'Password not the same'}),401
        joined_at = datetime.utcnow()
        user = User(username =username, email = email, password_hash = password)
        print('here')
        id = add_entry(user)
    except IntegrityError as e:
        raise CustomDatabaseError("This email/username is already in use.") from e
    except Exception as e:
        raise CustomApplicationError("An unexpected error occurred.") from e
    return jsonify(
        {
            "id": id,
            'username':username,
            "email": email,
            "password": password1,
            "joined_at": joined_at,
        }
    ) , 201

from werkzeug.security import check_password_hash
# Set the methods for api login
@app.route('/api/login', methods=['POST'])
def api_login():
    try: 
        data = request.get_json()

        email = data['email']
        password = data['password']
        user = User.query.filter_by(email=email).first()
        print(user)
        if user is None:
            return jsonify({"error": "User not found"}), 404
        # Here's the fix: use check_password_hash to compare the password correctly
        if not check_password_hash(user.password_hash, password) and user.password_hash != password:
            return jsonify({'error': 'Wrong Password'}), 403
        session["user_id"] = user.id
        return jsonify(
            {
                "id": user.id,
                "email": email,
                "message": "Login successful."
            }
        ), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "Database error occurred."}), 500


@app.route('/api/test_auth', methods=['GET'])
def test_auth():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True}), 200
    else:
        return jsonify({'authenticated': False}), 401