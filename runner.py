import struct
import os
from main import evaluate_model
import redis

def toRedis(a,n):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    r = redis.Redis(host='redis', port=6379, db=0)

    if isinstance(a, str):
        r.set(n,a)
        return
    h, w = a.shape
    shape = struct.pack('>II',h,w)
    encoded = shape + a.tobytes()

    # Store encoded data in Redis
    r.set(n,encoded)

if __name__ == '__main__':
    id = os.getenv('ID')
    filename = os.getenv('FILENAME')

    out = evaluate_model(f'images/{filename}')

    toRedis(out, id)
