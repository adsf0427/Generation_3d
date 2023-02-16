import objaverse
objaverse.__version__

uids = objaverse.load_uids()
len(uids), type(uids)

annotations = objaverse.load_annotations(uids[:10])
annotations

annotations = objaverse.load_annotations()
selected_uids =  [uid for uid, annotation in annotations.items() if 'car' in annotation["name"].split() ]
len(selected_uids)

import multiprocessing
processes = multiprocessing.cpu_count()
processes

import random

random.seed(427)

uids = objaverse.load_uids()
random_object_uids = random.sample(uids, 100)

random_object_uids

annotations['21f73626f9144689a7732915ff4bafb6']

objects = objaverse.load_objects(
    uids=selected_uids[:10],
    download_processes=processes
)
objects[selected_uids[0]]
annotations[selected_uids[0]]
names = [name for name in annotations.items()]