"""Util functions"""
import random

import torch
import numpy as np

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


'''CLASS_LABELS = {
    'Visceral': {
        'all': set([1,2,13,14,17,18]),
        0: set([2,13,14,17,18]),
        1: set([1,13,14,17,18]),
        2: set([1,2,17,18]),
        3: set([1,2,13,14]),
    }
}'''

CLASS_LABELS = {
    'Visceral': {
        'all': set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
        0: set([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
        1: set([1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
        2: set([1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,20]),
        3: set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20]),
    }
}


