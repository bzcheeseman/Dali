"""
Get music transcription dataset from:
http://c4dm.eecs.qmul.ac.uk/rdr/handle/123456789/13

E. Benetos, A. Klapuri, and S. Dixon,
"Score-informed transcription for automatic piano tutoring,"
in Proc. 20th European Signal Processing Conference, pp. 2153-2157, Aug. 2012.

"""
import random
import sys

from os import listdir, makedirs, stat
from os.path import isdir, isfile, join, dirname, realpath, exists

# add data to path
DATA_DIR = dirname(dirname(realpath(__file__)))
sys.path.append(DATA_DIR)
from utils import execute_bash, collect_files_with_ext
from midi.utils import midiread


from scipy.io import wavfile
import numpy as np

THIS_DATA_DIR  = dirname(realpath(__file__))
DOWNLOADED_ZIP = join(THIS_DATA_DIR, "dataset.zip")
DOWNLOADED_DIR = join(THIS_DATA_DIR, "dataset")
FILE_URL="http://c4dm.eecs.qmul.ac.uk/rdr/bitstream/handle/123456789/13/Score-informed%20Piano%20Transcription%20Dataset.zip?sequence=1"

if __name__ == '__main__':
    if not exists(DOWNLOADED_ZIP):
        execute_bash("wget -O {path} {url}".format(url=FILE_URL, path=DOWNLOADED_ZIP))
    if exists(DOWNLOADED_DIR) and isdir(DOWNLOADED_DIR):
        execute_bash("rm -rf %s" % (DOWNLOADED_DIR))
    execute_bash("rm %s " % (join(THIS_DATA_DIR, "*.npy")))
    makedirs(DOWNLOADED_DIR)
    execute_bash("unzip %s -d %s" % (DOWNLOADED_ZIP, DOWNLOADED_DIR))

    files = collect_files_with_ext(DOWNLOADED_DIR, ".wav")

    for subpath, name in files:
        if name.endswith(".wav") and "Chromatic" not in name:
            sampling_rate, music = wavfile.read(subpath)
            np.save(join(THIS_DATA_DIR, name.replace(".wav", ".npy")), music)
            piece = midiread(str(subpath).replace(".wav", "_correct.mid"))
            np.save(join(THIS_DATA_DIR, name.replace(".wav", ".mid.npy")), piece.piano_roll)

    execute_bash("rm -rf %s" % (DOWNLOADED_DIR))
    execute_bash("rm -rf %s" % (DOWNLOADED_ZIP))

