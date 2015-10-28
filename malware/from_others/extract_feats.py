import libarchive.public
import pandas as pd
import numpy as np
from multiprocessing import Pool, JoinableQueue, Queue

TRAIN_PATH = '../data/train.7z'


def count_bytes(lines):
    counts = np.zeros([257], dtype=np.int64)
    for l in lines:
        elems = l.split(' ')
        for el in elems[1:]:
            if (el == '??'):
                counts[256] += 1
            else:
                counts[np.int(el, 16)] += 1
    return counts


def get_features((q_in, q_out)):
    while True:
        (name, text) = q_in.get()

        lines = ''.join(text).split('\r\n')

        counts = count_bytes(lines)

        q_out.put([name, counts])
        q_in.task_done()

q = JoinableQueue(20)
q_feats = Queue()

pool = Pool(6, get_features, ((q, q_feats),))


with libarchive.public.file_reader(TRAIN_PATH) as archive:
    for entry in archive:

        # Use only .bytes
        if (entry.pathname.find('.bytes') != -1):
            text = []
            for b in entry.get_blocks():
                text.append(b)

            q.put((entry.pathname, text), True)

    q.close()
    q.join()
    pool.close()

# Now you can get a list of features like that
feats = []
for i in range(q_feats.qsize()):
    feats.append(q_feats.get())
