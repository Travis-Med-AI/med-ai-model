import struct
import os
from main import evaluate_model
import redis
import numpy as np
from skimage import io
import json
from functools import singledispatch
import numpy as np


def save_output(output, study_id):
    """Store given Numpy array 'a' in Redis under key 'n'"""

    r = redis.Redis(host='redis', port=6379, db=0)

    # Store encoded data in Redis
    r.set(study_id, json.dumps(output))

def save_images(images, file_paths):
    for image, path in zip(images, file_paths):
        io.imsave('{path}/output.jpg'.format(path=path), image)

def stringify_outputs(out):
    if out.class_probabilities:
        out.class_probabilities =  {k:float(v) for k, v in out.class_probabilities.items()}
    return  out.__dict__

if __name__ == '__main__':
    eval_id = os.getenv('ID')
    filenames = os.getenv('FILENAMES').split(',')

    print('filenames', filenames)
    file_paths = ['images/{name}'.format(name=name) for name in filenames]

    outputs = evaluate_model(file_paths)

    outputs = [stringify_outputs(output) for output in outputs]

    save_output(outputs, eval_id)
