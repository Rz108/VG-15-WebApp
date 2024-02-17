from application import app 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
if __name__ == '__main__':
    app.run(debug = True , port = 5000  , use_reloader = False)