# test_datasets.py
import pytest

import numpy as np
import torch

from sign_language_classifier.datasets import ASLAlphabet,ASLTestAlphabet

def test_ASLTestAlphabet_test_lastsample():
    """Check last item in the testset"""
    dset = ASLAlphabet()
    letter,sample = dset[len(dset)-1]
    assert(letter==28)
    assert(sample.shape==torch.zeros(3,200,200).shape)

def test_ASLAlphabet_train_lastsample():
    """Check last item in the trainset"""
    dset = ASLAlphabet(type="train")
    letter,sample = dset[len(dset)-1]
    assert(letter==28)
    assert(sample.shape==torch.zeros(3,200,200).shape)

def test_ASLAlphabet_val_lastsample():
    """Check last item in the valset"""
    dset = ASLAlphabet(type="val")
    letter,sample = dset[len(dset)-1]
    assert(letter==28)
    assert(sample.shape==torch.zeros(3,200,200).shape)
