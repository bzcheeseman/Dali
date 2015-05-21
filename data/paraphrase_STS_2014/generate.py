"""
SemEval 2014 data collection script:

Source is here: http://alt.qcri.org/semeval2014/task10/index.php?id=data-and-tools

"""
import random
import sys

from os import listdir, makedirs
from os.path import isdir, isfile, join, dirname, realpath, exists

# add data to path
DATA_DIR = dirname(dirname(realpath(__file__)))
sys.path.append(DATA_DIR)
from utils import print_progress, execute_bash

THIS_DATA_DIR = dirname(realpath(__file__))

ZIP_URL = "http://alt.qcri.org/semeval2014/task10/data/uploads/sts-en-gs-2014.zip"
ZIP_LOCAL = join(THIS_DATA_DIR, "dataset.zip")
UNZIPPED_LOCAL = join(THIS_DATA_DIR, "dataset")
TOKENIZED_FILE = join(THIS_DATA_DIR, "train.tokenized.tsv")

def is_dataset_input(fname):
    return fname.endswith(".txt") and ("readme" not in fname) and (".input." in fname)

def collect_files_with_ext(path, extension):
    paths = [(join(path, subpath), subpath) for subpath in listdir(path)]
    files = []
    for subpath, name in paths:
        if isdir(subpath):
            files += collect_files_with_ext(subpath, extension)
        else:
            if subpath.endswith(extension):
                files.append((subpath, name))
    return files

def collect_text_files(path):
    return [(subpath, name) for subpath, name in collect_files_with_ext(path, ".txt") if is_dataset_input(subpath)]

try:
    from xml_cleaner import to_raw_text_markupless
    tokenizer_available = True
except ImportError:
    print("""
Could not import xml_cleaner, not tokenization available.
Please install xml_cleaner:

    pip3 install xml_cleaner

""")
    tokenizer_available = False

def tokenize_sentences(text):
    sentences = text.strip().split("\t")
    gen_sentences = [" ".join(tsentence) for sentence in sentences for tsentence in to_raw_text_markupless(sentence)]
    return "\t".join(gen_sentences[0:2]) + " ".join(gen_sentences[2:])

if __name__ == '__main__':
    execute_bash("rm -rf %s" % (ZIP_LOCAL,))
    execute_bash("rm -rf %s" % (UNZIPPED_LOCAL,))
    execute_bash("wget -O {path} {url}".format(url=ZIP_URL, path=ZIP_LOCAL))
    execute_bash("unzip {zipfile} -d {target}".format(zipfile=ZIP_LOCAL, target=UNZIPPED_LOCAL))

    tar_files = collect_files_with_ext(UNZIPPED_LOCAL, ".tgz")
    for tar_file, tar_file_name in tar_files:
        execute_bash("tar -xf %s -C %s" % (tar_file, UNZIPPED_LOCAL))

    dataset_input_names = collect_text_files(UNZIPPED_LOCAL)

    with open(TOKENIZED_FILE, "wt")               as ftokenized:
        for dataset_fname, dataset_name in dataset_input_names:
            with open(join(THIS_DATA_DIR, dataset_name.replace(".txt", ".tsv")), "wt") as fout:
                with open(dataset_fname, "rt")                                         as finputs:
                    with open(dataset_fname.replace(".input.", ".gs."), "rt")          as flabels:
                        label_lines = (line for line in flabels)
                        input_lines = (line for line in finputs)

                        for label, sentences in zip(label_lines, input_lines):
                            fout.write(sentences.strip() + "\t" + label)
                            if tokenizer_available:
                                ftokenized.write(tokenize_sentences(sentences) + "\t" + label)


    if not tokenizer_available:
        execute_bash("rm %s" % (TOKENIZED_FILE,))
    execute_bash("rm -rf %s" % (UNZIPPED_LOCAL))
    execute_bash("rm -rf %s" % (ZIP_LOCAL))
