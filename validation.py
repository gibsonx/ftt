import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd

from tensorflow import reshape, nest, config
from tensorflow.keras import losses, metrics, optimizers
# Test the TFF is working:
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D
import nest_asyncio
nest_asyncio.apply()
import os
import secrets
import logging
import boto3
from botocore.client import Config,ClientError
import tarfile
import urllib3
import shutil
from tensorflow_federated.python.simulation import FileCheckpointManager

print(tf.__version__)
tff.federated_computation(lambda: 'Hello, World!')()

# seq = os.environ.get('seq')
# task_name = os.environ.get('task')
# bucket_name = str(os.environ.get('uuid'))

# if seq is None:
seq = 1
task_name = "gibson"
bucket_name = "beefa088-da23-4434-943f-12900ed84e86"
#
method = "tff_training"
client_lr = 1e-2
server_lr = 1e-2
split = 1
NUM_ROUNDS = 10
NUM_EPOCHS = 2
BATCH_SIZE = 1
PREFETCH_BUFFER = 10

data_path = '/opt/train/'

df_orig_train = pd.read_csv(data_path + 'mnist_train.csv')
df_orig_test = pd.read_csv(data_path + 'mnist_test.csv')
# print(df_orig_train.shape[0])
x_train = df_orig_train.iloc[:,1:].to_numpy().astype(np.float32).reshape(df_orig_train.shape[0],28,28,1)[:1]
y_train = df_orig_train.iloc[:,0].to_numpy().astype(np.int32).reshape(df_orig_train.shape[0],1)[:1]
x_test = df_orig_test.iloc[:,1:].to_numpy().astype(np.float32).reshape(df_orig_test.shape[0],28,28,1)[:10000]
y_test = df_orig_test.iloc[:,0].to_numpy().astype(np.int32).reshape(df_orig_test.shape[0],1)[:10000]

total_image_count = len(x_train)
image_per_set = int(np.floor(total_image_count/split))

client_train_dataset = collections.OrderedDict()
for i in range(1, split+1):
    client_name = "client_" + str(i)
    start = image_per_set * (i-1)
    end = image_per_set * i
    print(f"Adding data from {start} to {end} for client : {client_name}")
    data = collections.OrderedDict((('label', y_train[start:end]), ('pixels', x_train[start:end])))
    client_train_dataset[client_name] = data

train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

sample_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
sample_element = next(iter(sample_dataset))

SHUFFLE_BUFFER = image_per_set

def preprocess(dataset):
    def batch_format_fn(element):
        print(element['pixels'])
        return collections.OrderedDict(
            x=reshape(element['pixels'], [-1, 28, 28, 1]),
            y=reshape(element['label'], [-1, 1]))
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

preprocessed_sample_dataset = preprocess(sample_dataset)
sample_batch = nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_sample_dataset)))

def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

federated_train_data = make_federated_data(train_dataset, train_dataset.client_ids)
print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))

def create_keras_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation="relu",input_shape = (28,28,1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])
    return model

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_sample_dataset.element_spec,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()])

def create_bucket(bucket_name):
    response = s3_client.buckets.all()
    is_exist = False
    for bucket in response:
        if bucket.name == bucket_name:
            is_exist == True
    if is_exist == False:
        try:
            s3_client.create_bucket(Bucket=bucket_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True
    else:
        return False

def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    try:
        s3_client.Object(bucket, file_name).upload_file(object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def tardir(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                # print(dirs)
                tar_handle.add(os.path.join(root, file))

def untar(path):
    with tarfile.open(path) as tar_handle:
        tar_handle.extractall()
        tar_handle.close()

def DownloadObj(bucket,obj):
    try:
        s3_client.Bucket(bucket).download_file(obj, obj)
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
            return False
    return True

def download_dir(prefix, local, bucket, client):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)

def untardir(path):
    for path, directories, files in os.walk(path):
        for f in files:
            if f.endswith(".tar.gz"):
                tar = tarfile.open(os.path.join(path,f), 'r:gz')
                tar.extractall()
                tar.close()


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: optimizers.Adam(learning_rate=client_lr),
    server_optimizer_fn=lambda: optimizers.SGD(learning_rate=server_lr))


s3_client = boto3.client('s3', endpoint_url='http://192.168.1.104:9000',
                          aws_access_key_id='minio',
                          aws_secret_access_key='minio123'
                          )

download_dir('ckpt_', "./", bucket_name, client=s3_client)

untardir("./")

# tff_train_acc = []
tff_val_acc = []
# tff_train_loss = []
tff_val_loss = []

eval_model = None

# state.model.assign_weights_to(eval_model)
# path = '/content'
state_new = iterative_process.initialize()

ckpt_manager = FileCheckpointManager("./")
eval_model = None
for round_num in range(1,5):
    print(round_num)
    newstate = ckpt_manager.load_checkpoint(state_new,round_num=round_num)
    eval_model = create_keras_model()
    eval_model.compile(optimizer=optimizers.Adam(learning_rate=client_lr),
                      loss=losses.SparseCategoricalCrossentropy(),
                      metrics=[metrics.SparseCategoricalAccuracy()])
    newstate.model.assign_weights_to(eval_model)
    ev_result = eval_model.evaluate(x_test, y_test, verbose=0)
    tff_val_acc.append(ev_result[1])
    tff_val_loss.append(ev_result[0])
    print(f"Eval loss : {ev_result[0]} and Eval accuracy : {ev_result[1]}")

# newstate = ckpt_manager.load_checkpoint(state_new,round_num=9)
# eval_model = create_keras_model()
# eval_model.compile(optimizer=optimizers.Adam(learning_rate=client_lr),
#                   loss=losses.SparseCategoricalCrossentropy(),
#                   metrics=[metrics.SparseCategoricalAccuracy()])
# newstate.model.assign_weights_to(eval_model)
# ev_result = eval_model.evaluate(x_test, y_test, verbose=0)
# tff_val_acc.append(ev_result[1])
# tff_val_loss.append(ev_result[0])
# print(f"Eval loss : {ev_result[0]} and Eval accuracy : {ev_result[1]}")

metric_collection = {
                     "val_sparse_categorical_accuracy": tff_val_acc,
                     "val_loss": tff_val_loss}

print(metric_collection)
