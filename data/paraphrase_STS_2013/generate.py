import random
import sys

from os import listdir, makedirs
from os.path import isdir, isfile, join, dirname, realpath, exists

# add data to path
DATA_DIR = dirname(dirname(realpath(__file__)))
sys.path.append(DATA_DIR)
from utils import print_progress, execute_bash

THIS_DATA_DIR = dirname(realpath(__file__))

if __name__ == '__main__':
    print("""
        Data collection for SemEval 2013-2014-2015 is ongoing.
        See: http://alt.qcri.org/semeval2014/task10/index.php?id=data-and-tools
        for manual dataset downloading.
    """)
