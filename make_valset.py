"""

make_valset.py

Make the validation set by moving some images from /train to /val.

Note:
    - There is no testing file for this script

"""
import numpy as np
import os

def make_validation_set(traindir="../asl_alphabet/train",
                        valdir="../asl_alphabet/val",
                        val_frac=0.15):
    """
    make_validation_set

    Make the validation set by moving some images from traindir to valdir.
    Transfers np.floor(3000*val_frac) samples from each class into valdir. Uniformly samples images
    from class folders.

    inputs:
        traindir - (str) directory containing folders of train examples for the
                   the asl_dataset (traindir contains directories of class images)
        valdir - (str) directory to move validation images to
        val_frac - (float) fraction of training images to move to val folder. Equal number of samples per class.
    """
    if val_frac < 0 or val_frac > 1:
        raise ValueError("val_frac invalid! Please use 0 <= val_frac <= 1")

    if not os.path.exists(traindir):
        raise FileNotFoundError("Training set directory not found!")

    if os.path.exists(valdir):
        raise FileExistsError("Validation set directory already exists. Please clean up using python make_valset.py --undo")
    else:
        os.mkdir(valdir)
        make_empty_letterdirs(traindir=traindir, valdir=valdir)

    #
    # 29 class directories, 3000 samples per directory
    #
    # Loop through the classes and transfer np.floor(3000*val_frac) random images
    # to valdir for each.

    valSamplesPerClass = int(np.floor(3000*val_frac)) # train dataset originally has 3000 samples per class
    letters = os.listdir(traindir)
    letters.sort() # [ 'A', 'B', ...'space'...'Y', 'Z']

    for letter in letters:
        idx_list = np.random.choice(3000, valSamplesPerClass, replace=False)
        imNames = os.listdir(os.path.join(traindir, letter))
        for i in range(valSamplesPerClass):
            src = os.path.join(traindir, letter, imNames[idx_list[i]])
            dest = os.path.join(valdir, letter, imNames[idx_list[i]])
            os.rename(src, dest) # transfer file

    return valdir

def make_empty_letterdirs(traindir="../asl_alphabet/train",
                          valdir="../asl_alphabet/val"):
    """
    make_empty_letterdirs

    A function to create blank folders before transferring from train to val (in order to avoid "no such directory" error).
    Made specifically to be used with make_validation_set function.

    inputs:
        traindir - training data directory
        valdir - validation data directory
    """
    if not os.path.exists(traindir):
        raise FileNotFoundError("Training set directory not found!")

    letters = os.listdir(traindir)
    for letter in letters:
        os.mkdir(os.path.join(valdir, letter))

def merge_trainset_valset(traindir="../asl_alphabet/train",
                          valdir="../asl_alphabet/val"):
    """
    merge_trainset_valset

    A function to put validation samples back into the training directories.

    inputs:
        traindir - training data directory
        valdir - validation data directory
    """
    # make sure directories exist
    if not os.path.exists(traindir):
        raise FileNotFoundError("Training set directory not found!")
    elif not os.path.exists(valdir):
        raise FileNotFoundError("Validation set directory not found!")

    # define and sort letter directories
    trainletters = os.listdir(traindir)
    trainletters.sort()
    valletters = os.listdir(valdir)
    valletters.sort()
    for i in range(len(trainletters)):
        if trainletters[i] != valletters[i]:
            msg = "train and validation directories do not contain the same letters!"
            msg += "trainletters[i] = {}, valletters[i] = {}".format(trainletters[i], valletters[i])
            raise ValueError(msg)

    # move the samples
    for letter in valletters:
        letterPath = os.path.join(valdir, letter)
        imNames = os.listdir(letterPath)
        for name in imNames:
            src = os.path.join(valdir, letter, name)
            dest = os.path.join(traindir, letter, name)
            os.rename(src, dest)
        os.rmdir(letterPath)
    os.rmdir(valdir)

    for letter in os.listdir(traindir):
        if len(os.listdir(os.path.join(traindir, letter))) != 3000:
            raise Exception("Number of samples per letter is not 3000! [letter = {}]".format(letter))


# Run as script
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--traindir",type=str,default="../asl_alphabet/train",help="Directory containing training data folders")
    parser.add_argument("--valdir",type=str,default="../asl_alphabet/val",help="Directory containing validation data folders")
    parser.add_argument("--undo",action="store_true",help="Undo validation split by moving all validation samples back into the train directory")
    parser.add_argument("-vf","--valfrac",type=float,default=0.15,help="Fraction of dataset to move into the validation directory")
    opts = parser.parse_args()

    if not opts.undo:
        np.random.seed(0) # use for repeatability
        make_validation_set(traindir=opts.traindir, valdir=opts.valdir, val_frac=opts.valfrac)
    else:
        merge_trainset_valset(traindir=opts.traindir, valdir=opts.valdir)
