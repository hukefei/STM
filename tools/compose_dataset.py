import os
import random

davis_set_t = r'/workspace/tianchi/tianchiyusai/ImageSets/davis_train.txt'
davis_set_v = r'/workspace/tianchi/tianchiyusai/ImageSets/davis_val.txt'
tianchi_set_t = r'/workspace/tianchi/tianchiyusai/ImageSets/train.txt'
VAL_RATIO = 0.05

save_t = r'/workspace/tianchi/tianchiyusai/ImageSets/total_train.txt'
save_v = r'/workspace/tianchi/tianchiyusai/ImageSets/total_val.txt'

with open(tianchi_set_t, 'r') as f:
    tianchi_all = f.read().splitlines()

with open(davis_set_t, 'r') as f:
    davis_train = f.read().splitlines()

with open(davis_set_v, 'r') as f:
    davis_val = f.read().splitlines()

tianchi_val = random.sample(tianchi_all, int(VAL_RATIO * len(tianchi_all)))
tianchi_train = []
for i in tianchi_all:
    if i not in tianchi_val:
        tianchi_train.append(i)

total_train = tianchi_train + davis_train + davis_val
total_val = tianchi_val

with open(save_t, 'w') as f:
    for i in total_train:
        f.write(i)
        f.write('\r\n')

with open(save_v, 'w') as f:
    for i in total_val:
        f.write(i)
        f.write('\r\n')