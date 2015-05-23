import random
import sys

from os import listdir, makedirs, stat
from os.path import isdir, isfile, join, dirname, realpath, exists

# add data to path
DATA_DIR = dirname(dirname(realpath(__file__)))
sys.path.append(DATA_DIR)
from utils import print_progress, execute_bash

THIS_DATA_DIR = dirname(realpath(__file__))
DOWNLOADED_FILE = join(THIS_DATA_DIR, "wikianswers.paraphrases.tsv.gz")
FILE_URL="https://www.dropbox.com/s/td3ionbuj80hrkb/wikianswers.paraphrases.tsv.gz?dl=0"

if __name__ == '__main__':
    if exists(DOWNLOADED_FILE):
        print("Found file.")
    else:
        execute_bash("wget -O {path} {url}".format(url=DOWNLOADED_FILE, path=path))
        print("Downloaded file")

