"""
Tokenize Split
--------------

Small Utility for doing text tokenization and splitting into
training and validation text files with 1 sentence per line.

Uses `xml_cleaner`

"""
import argparse
import io
import os
import random, sys

from xml_cleaner import to_raw_text_markupless

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", help="Input file")
parser.add_argument("-t", "--train", help="Output training set")
parser.add_argument("-v", "--validation", help="Output validation set", default="")
parser.add_argument("-r", "--split_ratio", help="Train/Validation split", default=0.1, type=float)
parser.add_argument("-s", "--perform_split", help="should split be performed ?", default=True, type=bool)

args = parser.parse_args()


if __name__ == "__main__":
    total_size = os.stat(args.input).st_size
    i = 0
    with open(args.validation, "wt") as fvalid:
        with open(args.train, "wt") as fout:
            fin = open(args.input, "rb")
            text_fin = io.TextIOWrapper(fin, newline='')
            for line in text_fin:
                i+=1
                if i % 100 == 0:
                    progress = fin.tell() / total_size
                    print("█" * (int(20 * progress)) + " %.1f%% \r" % (100 * progress,), end="", flush=True)
                if args.perform_split: line = line.split("\t", 1)[1]
                for sentence in to_raw_text_markupless(line):
                    if random.random() > args.split_ratio:
                        fout.write(" ".join(sentence))
                        fout.write("\n")
                    else:
                        fvalid.write(" ".join(sentence))
                        fvalid.write("\n")
            fin.close()
