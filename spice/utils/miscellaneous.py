# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import json
import logging
import os
from .comm import is_main_process
import time
import torch

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_labels(dataset_list, output_dir):
    if is_main_process():
        logger = logging.getLogger(__name__)

        ids_to_labels = {}
        for dataset in dataset_list:
            if hasattr(dataset, 'categories'):
                ids_to_labels.update(dataset.categories)
            else:
                logger.warning("Dataset [{}] has no categories attribute, labels.json file won't be created".format(
                    dataset.__class__.__name__))

        if ids_to_labels:
            labels_file = os.path.join(output_dir, 'labels.json')
            logger.info("Saving labels mapping into {}".format(labels_file))
            with open(labels_file, 'w') as f:
                json.dump(ids_to_labels, f, indent=2)


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.text)

def choose_num_workers(generator, train_dataset, train_sampler, batch_size):
    last_time = 10e10
    chosen_num_worker = -1
    for num_workers in range(2, 21, 4):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=num_workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True, generator=generator, persistent_workers=True)

        for i, data in enumerate(train_loader, 0):
            break

        start = time.time()
        for i, data in enumerate(train_loader, 0):
            if i > 30:
                break

        end = time.time()
        print(f'time:{end-start}, workers:{num_workers}')
        train_loader = None
        del train_loader

        if last_time < (end - start):
            chosen_num_worker = num_workers - 4
            break
        else:
            last_time = end - start

    if chosen_num_worker == -1:
        chosen_num_worker = 20

    return chosen_num_worker