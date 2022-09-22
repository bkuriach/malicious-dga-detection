import json
import joblib
import numpy as np
from azureml.core.model import Model
from azureml.core import Workspace

import tensorflow_hub as hub
import numpy as np
import pandas as pd
import keras
# from keras.models import ModelModel
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Lambda
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from keras.utils import to_categorical
import os
from train import embed, build_model

# embed = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

# def ELMoEmbedding(x):    
#     return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

   
# ws = Workspace.from_config()
# for model in Model.list(ws):
#     print(model.name, ":", model.version)

# Called when the service is loaded
def init():
    global loaded_model
    model_path_json = Model.get_model_path(model_name='elmo-model.json',version=7)
    print("model_path_json", model_path_json)
    model_path_weights = Model.get_model_path(model_name='elmo-model-weights.h5',version=7)
    print("model_path_weights", model_path_weights)
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    # model_path_json = os.path.join(
    #     os.getenv("AZUREML_MODEL_DIR"), "outputs/elmo-model.json"
    # )
    # model_path_weights = os.path.join(
    #     os.getenv("AZUREML_MODEL_DIR"), "outputs/elmo-model-weights.h5"
    # )

    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        # json_file = open('outputs/model.json', 'r')
        json_file = open(model_path_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        print("Loaded model json from disk")    
        # evaluate loaded model on test data
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_path_weights)
        print("Loaded model weights from disk")    
        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    print("Model Loaded")

# Called when a request is received
def run(raw_data):
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("Inside run")
        ip = [raw_data]
        ip.append('')
        ip = np.array(ip,ndmin=1,dtype=object)
        predictions = loaded_model.predict(ip)
        print("Prediction :",predictions)

        # # Return the predictions as any JSON serializable format
        return json.dumps(predictions.tolist()[0])

