#!/usr/bin/env python

"""Functions for downloading and reading MNIST data."""
import gzip
import os
from six.moves.urllib.request import urlretrieve
import numpy
SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"

def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print ('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

