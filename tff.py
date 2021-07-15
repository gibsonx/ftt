import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
from tensorflow_federated.python.simulation import FileCheckpointManager
from tensorflow import reshape, nest, config
from tensorflow.keras import losses, metrics, optimizers
# Test the TFF is working:
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import nest_asyncio
nest_asyncio.apply()
import os
import secrets
import logging
import boto3
from botocore.client import Config,ClientError
import tarfile
import urllib3

tff.federated_computation(lambda: 'Hello, World!')()
print(tf.__version__)


seq = os.environ.get('seq')
task_name = os.environ.get('task')
bucket_name = str(os.environ.get('uuid'))
# seq = 1
# task_name = "test"
# bucket_name = secrets.token_hex(nbytes=12)

method = "tff_training"
client_lr = 1e-2
server_lr = 1e-2
split = 10
NUM_ROUNDS = 5
NUM_EPOCHS = 5
BATCH_SIZE = 1
PREFETCH_BUFFER = 10

root_path = '/opt/train/'


df_orig_train = pd.read_csv(root_path + 'data.csv')
# df_orig_test = pd.read_csv('mnist_test.csv')
print(df_orig_train.shape[0])
x_train = df_orig_train.iloc[:,1:].to_numpy().astype(np.float32).reshape(df_orig_train.shape[0],28,28,1)[:10]
y_train = df_orig_train.iloc[:,0].to_numpy().astype(np.int32).reshape(df_orig_train.shape[0],1)[:10]
# x_test = df_orig_test.iloc[:,1:].to_numpy().astype(np.float32).reshape(9999,28,28,1)[:10]
# y_test = df_orig_test.iloc[:,0].to_numpy().astype(np.int32).reshape(9999,1)[:10]

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
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
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
        Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)),
        Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',activation ='relu'),
        MaxPool2D(pool_size=(2,2)),
        Dropout(0.25),
        Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'),
        Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'),
        MaxPool2D(pool_size=(2,2), strides=(2,2)),
        Dropout(0.3),
        Flatten(),
        Dense(512, activation = "relu", use_bias= True),
        Dropout(0.5),
        Dense(10, activation = "softmax")
    ])
    return model

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_sample_dataset.element_spec,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: optimizers.Adam(learning_rate=client_lr),
    server_optimizer_fn=lambda: optimizers.SGD(learning_rate=server_lr))

eval_model = None

state = iterative_process.initialize()

eval_model = None
for round_num in range(1, NUM_ROUNDS+1):
    state, tff_metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, tff_metrics))

filePath = root_path + str(bucket_name) + '/'

ckpt_manager = FileCheckpointManager(filePath)
ckpt_manager.save_checkpoint(state,round_num=int(seq))

objFolder = 'ckpt_'+ str(seq)
obj = 'ckpt_'+ str(seq) + '.tar.gz'

tarPath =  filePath + objFolder
tarFile = filePath  + obj

def tardir(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))

tardir(tarPath, tarFile)

s3_client = boto3.resource('s3',
                           endpoint_url='http://192.168.1.104:9000',
                           aws_access_key_id='minio',
                           aws_secret_access_key='minio123'
                           )

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

create_bucket(bucket_name)

upload_file(obj, bucket=bucket_name, object_name=tarFile)

http = urllib3.PoolManager()
resp = http.request(
    "PUT",
    "http://192.168.1.104:8000/task/" + task_name + '/',
    fields={
        "status": 1,
        "output": obj
    }
)
print(resp.data)
