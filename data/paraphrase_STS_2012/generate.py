import random
import sys

from os import listdir, makedirs
from os.path import isdir, isfile, join, dirname, realpath, exists

# add data to path
DATA_DIR = dirname(dirname(realpath(__file__)))
sys.path.append(DATA_DIR)
from utils import print_progress, execute_bash

THIS_DATA_DIR = dirname(realpath(__file__))

base_url = "http://www.cs.york.ac.uk/semeval-2012/task6/data/uploads/datasets/"

urls = [
    (base_url + "trial.tgz",     join(THIS_DATA_DIR, "trial.tgz"),     join(THIS_DATA_DIR, "trial"),     join(THIS_DATA_DIR, "dev")),
    (base_url + "train.tgz",     join(THIS_DATA_DIR, "train.tgz"),     join(THIS_DATA_DIR, "train"),     join(THIS_DATA_DIR, "train")),
    (base_url + "test-gold.tgz", join(THIS_DATA_DIR, "test-gold.tgz"), join(THIS_DATA_DIR, "test-gold"), join(THIS_DATA_DIR, "test")),
]

temp_dir = join(THIS_DATA_DIR, "tmp")

def delete_paths(paths):
    for path in paths:
        execute_bash('rm -rf %s' % (path,))

if __name__ == '__main__':
    if not exists(temp_dir):
        makedirs(temp_dir)
    # delete existing downloads
    delete_paths([out for url, path, out, newdir in urls] + [path for url, path, out, newdir in urls] + [newdir for url, path, out, newdir in urls])
    # download new ones
    for url, path, out, newdir in urls:
        execute_bash("wget -O {path} {url}".format(url=url, path=path))
        execute_bash("tar -xf %s -C %s" % (path, THIS_DATA_DIR))
        execute_bash("rm %s"            % join(out, "00-readme.txt"))
        execute_bash("mv {dataset_files} {tmp_dir}".format(
            tmp_dir=temp_dir,
            dataset_files=join(out, "*.txt"))
        )
        delete_paths([out])
        if not exists(newdir):
            makedirs(newdir)
        execute_bash("mv {tmp_files} {newdir}".format(
            tmp_files = join(temp_dir, "*.txt"),
            newdir = newdir)
        )
    delete_paths([temp_dir] + [path for url, path, out, newdir in urls])


