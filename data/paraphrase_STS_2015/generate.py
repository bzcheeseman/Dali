"""
Generate STS 2015 dataset.
Currently encrypted until dataset
is officially released.
"""
import random
import sys

from os import listdir, makedirs, stat
from os.path import isdir, isfile, join, dirname, realpath, exists

# add data to path
DATA_DIR = dirname(dirname(realpath(__file__)))
sys.path.append(DATA_DIR)
from utils import print_progress, execute_bash

THIS_DATA_DIR = dirname(realpath(__file__))

ENCRYPT_SCRIPT = join(THIS_DATA_DIR, "generate.sh")
ARCHIVE_DIR = join(THIS_DATA_DIR, "secret")

def boldify(text):
    return "\033[1m%s\033[0m" % (text,)

def greenify(text):
    return "\033[32m%s\033[0m" % (text,)

def yellowify(text):
    return "\033[33m%s\033[0m" % (text,)

def encrypt_archive(archive_name, directory_name):
    if exists(archive_name) and isfile(archive_name):
        filesize = stat(archive_name).st_size
        if filesize < 100:
            print(yellowify("Present archive has wrong size (%d bytes). Removing." % (filesize,)))
            execute_bash("rm %s" % (archive_name,))

    if not exists(archive_name) or not isfile(archive_name):
        if exists(directory_name) and isdir(directory_name):
            print(boldify("Encrypting data"))
            execute_bash("sh %s encrypt" % (ENCRYPT_SCRIPT))
            print(greenify("Successfully Encrypted data"))

def decrypt_archive():
    print(boldify("Decrypting data"))
    execute_bash("sh %s decrypt" % (ENCRYPT_SCRIPT))

def ensure_decrypted(directory_name):
    print(yellowify("Checking decrypted data"))
    for path in ["dev.tsv", "README.md", "test.label.tsv", "test.tsv", "train.tsv"]:
        assert(exists(join(directory_name, path)) and isfile(join(directory_name, path))), "Could not find %s in decrypted archive" % (path,)
    print(greenify("Successfully Decrypted data"))

if __name__ == '__main__':
    encrypt_archive(join(THIS_DATA_DIR, "private.gpg"), ARCHIVE_DIR)
    decrypt_archive()
    ensure_decrypted(ARCHIVE_DIR)
