import pytest

import torch

from sign_language_classifier.datasets import ASLAlphabet
from sign_language_classifier.networks import Flatten,SLClassifier

@pytest.fixture
def net():
    return SLClassifier()

@pytest.fixture
def dset():
    return ASLAlphabet()

def test_flatten():
    """Test the Flatten layer"""
    x = torch.zeros(1,2,3,4,5)
    FL = Flatten()
    assert(FL(x).shape==x.flatten(start_dim=1).shape)
    FL = Flatten(start_dim=3)
    assert(FL(x).shape==x.flatten(start_dim=3).shape)

def test_SLClassifier_backprop(net,dset):
    """Test that backprop works on the summed output"""
    letter,sample = dset[0]
    y = net(sample.unsqueeze(0))
    y.sum().backward()

def test_SLClassifier_evalmode(net,dset):
    """Test SLClassifier eval mode"""
    net.eval()

    letter,sample = dset[0]
    scores = net(sample.unsqueeze(0))
    assert(scores.shape==torch.zeros(1,29).shape)

def test_SLClassifier_single(net,dset):
    """Test the SLClassifier on a small batch input"""
    batchsize = 4
    samples = torch.zeros(batchsize,3,200,200)
    for i in range(batchsize):
        letter,sample = dset[i]
        samples[i] = sample

    y = net(samples)
    assert(y.shape==torch.zeros(4,29).shape)
