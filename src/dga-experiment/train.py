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
    domain_df = domain_df.sample(frac=0.01)
    domain_df.to_csv('sample_df.csv')
    run.upload_file(name = "outputs/domain_df.csv", path_or_stream = "sample_df.csv")

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
        history = model_elmo.fit(list_stentences_train, y_binary , epochs=1, batch_size =2048, validation_split = 0.2)
        model_elmo.save_weights('outputs/model_elmo_weights.h5')
        model_elmo.save('outputs/elmo_model.h5')
        # model_elmo.save('outputs/my_model')

        # serialize model to JSON
        model_json = model_elmo.to_json()
        with open("outputs/model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model_elmo.save_weights("outputs/model.h5")
        print("Saved model to disk")

    run.upload_file(name = "outputs/model.json", path_or_stream = "outputs/model.json")
    run.upload_file(name = "outputs/model.h5", path_or_stream = "outputs/model.h5")

    # with tf.Session() as session:
    #     K.set_session(session)
    #     session.run(tf.global_variables_initializer())
    #     session.run(tf.tables_initializer())
    #     print("predictions", model_elmo.predict(list_stentences_train[0:100]))

    # joblib.dump(value=model_elmo, filename="outputs/"+ "elmo_model1.h5")
    joblib.dump(value=domain_df, filename="sample_df.csv")

    # run.upload_file(name = "outputs/"+"elmo_model.pkl", path_or_stream = "outputs/elmo_model.pkl")

    run.upload_file(name = "outputs/model_elmo_weights.h5", path_or_stream = "outputs/model_elmo_weights.h5")
    run.upload_file(name = "outputs/elmo_model.h5", path_or_stream = "outputs/elmo_model.h5")

    # run.upload_file(name = "outputs/elmo_model1.h5", path_or_stream = "outputs/elmo_model1.h5")
    # run.upload_file(name = "outputs/my_model", path_or_stream = "outputs/my_model")


    run.complete()
    run.register_model(model_path="outputs/model_elmo_weights.h5", model_name='elmo_model_weights',
                    tags={'Training context':'Inline Training'})
    run.register_model(model_path="outputs/elmo_model.h5", model_name='elmo_model',
    tags={'Training context':'Inline Training'})

    # run.register_model(model_path="outputs/my_model", model_name='my_model',
    # tags={'Training context':'Inline Training'})


    # Download the model from run history
    # run.download_file(name='outputs/elmo_model.h5', output_file_path='outputs/elmo_model.h5')
    # run.download_file(name="outputs/elmo_model.h5",output_file_path=".")



    # import pickle
    # with open('elmo.pkl','w') as f:
    #     pickle.dumps(model_elmo)

    # service_name = 'elmo-service'
    # from azureml.core import Workspace

    # subscription_id = "4c7cb17d-ccf9-4af5-964f-b453c96bbd72"
    # resource_group = "aml-resources-bk"
    # workspace_name = "aml-workspace-bk"
    # ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)

    # from azureml.core import Model
    # service = Model.deploy(ws, service_name, [model_elmo], overwrite=True)
    # service.wait_for_deployment(show_output=True)


    # # load json and create model
    # json_file = open('outputs/model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = keras.models.model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("outputs/model.h5")
    # print("Loaded model from disk")
    
    # # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        json_file = open('outputs/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("outputs/model.h5")
        print("Loaded model from disk")
        
        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # print("predictions", loaded_model.predict(list_stentences_train[0:100]))

    run.register_model(model_path="outputs/model.h5", model_name='elmo-model-weights.h5',
                    tags={'Training context':'Inline Training'})
    run.register_model(model_path="outputs/model.json", model_name='elmo-model.json',
    tags={'Training context':'Inline Training'})

if __name__=="__main__":
    print("Training")
    run_train()