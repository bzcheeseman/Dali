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
parser.add_argument("-i", "--input", help="Input file", required = True)
parser.add_argument("-t", "--train", help="Output training set", required = True)
parser.add_argument("-v", "--validation", help="Output validation set", default="", required = True)
parser.add_argument("-r", "--split_ratio", help="Train/Validation split", default=0.1, type=float)
parser.add_argument("-s", "--perform_split", help="should split be performed ?", default=False, type=bool)
parser.add_argument("-k", "--keep_split", help="Should the elements before the tab character be kept ?", default=False, type=bool)
args = parser.parse_args()

def tokenize_and_write(file, text, token):
    for sentence in to_raw_text_markupless(text):
        file.write(" ".join(sentence))
        file.write(token)

if __name__ == "__main__":
    total_size = os.stat(args.input).st_size
    i = 0
    with open(args.validaton, "wt") as fvalid:
        with open(args.train, "wt") as fout:
            with open(args.input, "rb") as fin:
                text_fin = io.TextIOWrapper(fin, newline='')
                for line in text_fin:
                    i+=1
                    if i % 100 == 0:
                        progress = fin.tell() / total_size
                        print("█" * (int(20 * progress)) + " %.1f%% \r" % (100 * progress,), end="", flush=True)
                    to_train = random.random() > args.split_ratio
                    if args.perform_split and not args.keep_split:
                        line = line.split("\t", 1)[1]
                    elif args.perform_split and args.keep_split:
                        line = line.split("\t", 1)
                        tokenize_and_write(fout if to_train else fvalid, line[0], " ")
                        if to_train: fout.write("\t")
                        else: fvalid.write("\t")
                        tokenize_and_write(fout if to_train else fvalid, line[1], " ")
                        if to_train: fout.write("\n")
                        else: fvalid.write("\n")
                        continue
                    tokenize_and_write(fout if to_train else fvalid, line, "\n")
