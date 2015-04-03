import random
import sys

from os import listdir
from os.path import isdir, isfile, join, dirname, realpath

# add data to path
DATA_DIR = dirname(dirname(realpath(__file__)))
sys.path.append(DATA_DIR)
from utils import print_progress, execute_bash

THIS_DATA_DIR = dirname(realpath(__file__))

# important
ZIP_URL = 'http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
ZIP_LOCAL = join(THIS_DATA_DIR, 'trainDevTestTrees_PTB.zip')

def delete_paths(paths):
    for path in paths:
        execute_bash('rm -rf %s' % (path,))


if __name__ == '__main__':
    local_files = [join(THIS_DATA_DIR, f) for f in ['train.txt', 'test.txt', 'dev.txt']]
    delete_paths([ZIP_LOCAL, join(THIS_DATA_DIR, "trees")] + local_files)
    execute_bash('wget -O %s %s' % (ZIP_LOCAL, ZIP_URL))
    execute_bash('unzip %s -d %s' % (ZIP_LOCAL, THIS_DATA_DIR))
    execute_bash('mv %s %s' % (join(THIS_DATA_DIR, "trees", "*"), THIS_DATA_DIR))
    delete_paths([ZIP_LOCAL,join(THIS_DATA_DIR, "trees")])
