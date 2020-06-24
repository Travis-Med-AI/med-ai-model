import struct
import os
from main import evaluate_model
import redis
import numpy as np
from skimage import io

def save_output(output, study_id):
    """Store given Numpy array 'a' in Redis under key 'n'"""

    r = redis.Redis(host='redis', port=6379, db=0)

    save_location =  '/opt/images'
    volume_location = f'/{study_id}.npy'

    np.save(f'{save_location}{volume_location}', output)

    # Store encoded data in Redis
    r.set(study_id, volume_location)

def save_images(images, file_paths):
    for image, path in zip(images, file_paths):
        io.imsave(f'{path}/output.jpg', image)

if __name__ == '__main__':
    eval_id = os.getenv('ID')
    filenames = os.getenv('FILENAMES').split(',')
    save_image = bool(os.getenv('SAVE_IMAGE'))

    print('filenames', filenames)
    file_paths = [f'images/{name}' for name in filenames]

    out, images = evaluate_model(file_paths)
    if save_image:
        save_images(images, file_paths)
    save_output(out, eval_id)
