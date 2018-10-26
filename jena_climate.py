# coding: utf-8

import os
import numpy as np

# 载入数据
def load_data(filename):
    data = open(filename, 'r', encoding='utf-8').read()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values


    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std
    
    return float_data


def generator(data, lookback, delay, min_index, max_index, 
              shuffle=True, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
            
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
        
        
def data_gen(float_data):   
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    train_gen = generator(float_data, 
                          lookback=lookback, 
                          delay=delay, 
                          min_index=0, 
                          max_index=200000,
                          shuffle=True,
                          step=step, 
                          batch_size=batch_size)
    val_gen = generator(float_data, 
                        lookback=lookback, 
                        delay=delay, 
                        min_index=200001, 
                        max_index=300000,
                        step=step, 
                        batch_size=batch_size)
    test_gen = generator(float_data, 
                         lookback=lookback, 
                         delay=delay, 
                         min_index=300001, 
                         max_index=None,
                         step=step, 
                         batch_size=batch_size)
    
    return train_gen, val_gen, test_gen