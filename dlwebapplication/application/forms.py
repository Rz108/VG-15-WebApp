from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, SelectField, PasswordField, BooleanField, StringField
from wtforms.fields import DateField, EmailField, TelField
from wtforms.validators import Length, InputRequired, ValidationError, NumberRange,Length, Optional, URL, Email, Regexp, EqualTo,  ValidationError
import pandas as pd
import os


# Create the registration form for user to register
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators = [InputRequired()])
    email = EmailField('Email Address', validators =[InputRequired(), Email()])
    password_1 = PasswordField('Password', validators = [InputRequired()])
    password = PasswordField('Confirm Password', validators = [InputRequired(),EqualTo('password_1') ])
    submit = SubmitField('Register')

# Create the login form for the user to login
class Login(FlaskForm):
    email = EmailField('Email Address', validators =[InputRequired(), Email()])
    password = PasswordField('Password', validators =[ InputRequired()])
    save = BooleanField('Remeber Me', validators = [Optional()], default=False)
    submit = SubmitField('Login')


# Set the current directory to get the brands
current_directory = os.path.dirname(os.path.abspath(__file__))



# Creating the reset password form 
class ResetPasswordForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()])
    new_pw = PasswordField('New Password', validators=[InputRequired()])
    submit = SubmitField('Reset Password')


# Set the change password here
class ChangePasswordForm(FlaskForm):
    new_password = PasswordField('New Password', validators=[InputRequired()])
    confirm_new_password = PasswordField('Confirm New Password',
                                         validators=[InputRequired(),
                                                     EqualTo('new_password', message='Passwords must match.')])
    submit = SubmitField('Change Password')