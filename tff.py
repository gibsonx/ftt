import collections
import nest_asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import reshape, nest
# Test the TFF is working:
from tensorflow.keras import Sequential
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

nest_asyncio.apply()
import os
import logging
import boto3
from botocore.client import ClientError
import tarfile
import urllib3
from tensorflow_federated.python.simulation import FileCheckpointManager

print(tf.__version__)
tff.federated_computation(lambda: 'Hello, World!')()

#S3_storage##
endpoint = 'http://192.168.1.104:9000'
access_key = 'minio'
access_secret = 'minio123'

#Scheduler
scheduler_url = 'http://192.168.1.104:8000'

seq = int(os.environ.get('seq'))
task_name = os.environ.get('task')
bucket_name = str(os.environ.get('uuid'))

#
# seq = 1
# task_name = "gibson"
# bucket_name = "16be5162-eb3f-11eb-9a03-0242ac130007"
#
method = "tff_training"
client_lr = 1e-2
server_lr = 1e-2
split = 1
NUM_ROUNDS = 20
NUM_EPOCHS = 1
BATCH_SIZE = 1
PREFETCH_BUFFER = 10

data_path = '/opt/train/'

df_orig_train = pd.read_csv(data_path + 'data.csv')
# df_orig_test = pd.read_csv('mnist_test.csv')
print(df_orig_train.shape[0])
x_train = df_orig_train.iloc[:,1:].to_numpy().astype(np.float32).reshape(df_orig_train.shape[0],28,28,1)
y_train = df_orig_train.iloc[:,0].to_numpy().astype(np.int32).reshape(df_orig_train.shape[0],1)
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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar_handle)
        tar_handle.close()

def DownloadObj(bucket,obj):
    try:
        s3_client.Bucket(bucket).download_file(obj, obj)
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
            return False
    return True

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: optimizers.Adam(learning_rate=client_lr),
    server_optimizer_fn=lambda: optimizers.SGD(learning_rate=server_lr))


s3_client = boto3.resource('s3',
                           endpoint_url=endpoint,
                           aws_access_key_id=access_key,
                           aws_secret_access_key=access_secret
                           )

CurrentPath = os.getcwd()

oldObj = 'ckpt_'+ str(seq -1) + '.tar.gz'
bucket = create_bucket(bucket_name)
down = DownloadObj(bucket_name,oldObj)

state = iterative_process.initialize()

ckpt_manager = FileCheckpointManager(CurrentPath)

if down ==True:
    untar(oldObj)
    state = ckpt_manager.load_checkpoint(state,round_num=int(seq - 1))
    print("parameters is imported")
else:
    print("no model parameters is donwloaded")

for round_num in range(1, NUM_ROUNDS+1):
    state, tff_metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, tff_metrics))

ckpt_manager.save_checkpoint(state,round_num=int(seq))

objFolder = 'ckpt_'+ str(seq)
obj = 'ckpt_'+ str(seq) + '.tar.gz'

tardir(objFolder, obj)

upload_file(obj, bucket=bucket_name, object_name=obj)

http = urllib3.PoolManager()
resp = http.request(
    "PUT",
    scheduler_url + '/task/' + task_name + '/',
    fields={
        "status": 1,
        "output": obj
    }
)
print(resp.data)
resp = http.request(
    "GET",
    scheduler_url + '/jobrun/' + bucket_name + '/'
)
print(resp.data)