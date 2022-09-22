import os
import argparse
from azureml.core import Run, Dataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

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


embed = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

def ELMoEmbedding(x):
    global embed 
    import tensorflow_hub as hub 
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    embed = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)  
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def build_model():
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(256, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001))(embedding)
    pred = Dense(2, activation='softmax')(dense)
    model = keras.models.Model(inputs=[input_text], outputs = pred)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def run_train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
    parser.add_argument("--input-data-dga", type=str, dest='dga_dataset_id', help='dga dataset')
    parser.add_argument("--input-data-clean", type=str, dest='clean_dataset_id', help='clean dataset')
    args = parser.parse_args()

    # Set regularization hyperparameter (passed as an argument to the script)
    reg = args.reg_rate

    # Get the experiment run context
    run = Run.get_context()

    # Get the training dataset
    print("Loading Data...")
    malicious_dga_df = run.input_datasets['dga'].to_pandas_dataframe()[['domain', 'class']]
    clean_domains_df = run.input_datasets['clean_domains'].to_pandas_dataframe()[['domain', 'class']]

    domain_df = pd.concat([malicious_dga_df,clean_domains_df])
    domain_df = domain_df.sample(frac=1)
 
    print(malicious_dga_df.shape, malicious_dga_df.head())
    print(clean_domains_df.shape, clean_domains_df.head())
    print(domain_df.head())

    list_stentences_train = domain_df['domain'].fillna("CVxTz").values
    y = domain_df['class'].values
    y_binary = [1 if x =='dga' else 0 for x in y]
    y_binary = to_categorical(y_binary)
    print(y_binary)

    model_elmo = build_model()


    os.makedirs('outputs', exist_ok=True)
    print("Entering Model Training")
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        history = model_elmo.fit(list_stentences_train, y_binary , epochs=5, batch_size =1024, validation_split = 0.2)
        
        # serialize model to JSON
        model_json = model_elmo.to_json()
        with open("outputs/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model_elmo.save_weights("outputs/model.h5")
        print("Saved model to disk")

    run.upload_file(name = "outputs/model.json", path_or_stream = "outputs/model.json")
    run.upload_file(name = "outputs/model.h5", path_or_stream = "outputs/model.h5")

    
    run.complete()
    
    run.register_model(model_path="outputs/model.h5", model_name='elmo-model-weights.h5',
                tags={'Training context':'Inline Training'})
    run.register_model(model_path="outputs/model.json", model_name='elmo-model.json',
    tags={'Training context':'Inline Training'})

if __name__=="__main__":
    print("Training")
    run_train()