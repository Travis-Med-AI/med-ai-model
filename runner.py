import struct
import os

from pika import connection
from main import evaluate_model, load_model
import numpy as np
import skimage
import json
from functools import singledispatch
import numpy as np
from contextlib import redirect_stdout
import pika, sys, os, io
import traceback


rabbitmq_url = os.getenv('RABBITMQ_URL')

model = load_model()

def get_rabbit_connection():
    if rabbitmq_url:
        connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
    else:
        connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    return connection

def save_output(queue, ids, outputs, type='SAVE'):

    connection = get_rabbit_connection()
    channel = connection.channel()
    for id, output in zip (ids, outputs):
        message = {
            'id': id,
            'output': output,
            'type': type
        }

        channel.basic_publish(exchange='',
                            routing_key=queue,
                            body=json.dumps(message))
        

def save_images(images, file_paths):
    for image, path in zip(images, file_paths):
        skimage.io.imsave('{path}/output.jpg'.format(path=path), image)

def stringify_outputs(out):
    if out.class_probabilities:
        out.class_probabilities =  {k:float(v) for k, v in out.class_probabilities.items()}
    return  out.__dict__

def queue_callback(ch, method, properties, body):
    print(f"Received {body}")
    message = json.loads(body)
    task_type = message['type']

    if task_type == 'STOP':
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    files = message['files']
    files = ['images/{name}'.format(name=name) for name in files]

    ids = message['ids']
    do_inference(files, 'eval_results', ids, model)

def queue_setup(queue):
    connection = get_rabbit_connection()
    
    channel = connection.channel()
    channel.queue_declare(queue=queue)

    channel.basic_consume(queue=queue, on_message_callback=queue_callback, auto_ack=True)
    channel.start_consuming()




def do_inference(files, result_queue, ids, eval_model, log_queue='log_results'):
    with io.StringIO() as buf, redirect_stdout(buf):
        tb = ''
        try:

            outputs = evaluate_model(files, eval_model)
            outputs = [stringify_outputs(output) for output in outputs]
            save_output(result_queue, ids, outputs)
            logs = [buf.getvalue() for id in ids]
            save_output(log_queue, ids, logs ,'LOG')
        except :
            save_output(result_queue, ids, range(len(ids)), 'FAIL')
            tb = traceback.format_exc()
            logs = [buf.getvalue() for id in ids]
            save_output(log_queue, ids, logs ,'LOG')
            raise
            




if __name__ == '__main__':
    eval_id = os.getenv('ID')
    db_ids = os.getenv('DB_IDs')
    result_queue = os.getenv('RESULT_QUEUE')
    queue = os.getenv('QUEUE')
    run_single = os.getenv('RUN_SINGLE')
    filenames = os.getenv('FILENAMES')

    if filenames:
        filenames = os.getenv('FILENAMES').split(',')

    if(db_ids):
        db_ids = db_ids.split(',')
    
    if run_single:
        run_single = bool(run_single)

    print(f'filenames: {filenames}')
    print(f'queue: {queue}')
    print(f'result queue: {result_queue}')
    print(f'eval_id: {eval_id}')
    print(f'run_single: {run_single}, {os.getenv("RUN_SINGLE")}')
    print(f'db_ids: {db_ids}')

    if run_single:
        file_paths = ['images/{name}'.format(name=name) for name in filenames]
        model = load_model()
        do_inference(file_paths, result_queue, db_ids, model)
    else:
        queue_setup(queue)
