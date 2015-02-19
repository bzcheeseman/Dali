"""
Tokenize Split
--------------

Small Utility for doing text tokenization and splitting into
training and validation text files with 1 sentence per line.

Uses `xml_cleaner`

"""

from xml_cleaner import to_raw_text_markupless
import random, sys
import os
import io

def clean_and_output(path, outpath, valid_path, valid_percent = 0.1, perform_split = False):
    total_size = os.stat(path).st_size
    i = 0
    with open(valid_path, "wt") as fvalid:
        with open(outpath, "wt") as fout:
            fin = open(path, "rb")
            text_fin = io.TextIOWrapper(fin, newline='')
            for line in text_fin:
                i+=1
                if i % 100 == 0:
                    progress = fin.tell() / total_size
                    print("█" * (int(20 * progress)) + " %.1f%% \r" % (100 * progress,), end="", flush=True)
                if perform_split: line = line.split("\t", 1)[1]
                for sentence in to_raw_text_markupless(line):
                    if random.random() > valid_percent:
                        fout.write(" ".join(sentence))
                        fout.write("\n")
                    else:
                        fvalid.write(" ".join(sentence))
                        fvalid.write("\n")
            fin.close()

usage = """
{0} [input_file] [output_train_file] [output_validation_file] [(optional) validation percentage] [(optional) keep only elements after tab char]
"""

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print(usage.format(sys.argv[0]))
    else:
        path = sys.argv[1]
        outpath = sys.argv[2]
        valid_path = sys.argv[3]
        perform_split = sys.argv[5] == "1" if len(sys.argv) > 5  else False
        valid_percent = float(sys.argv[4]) if len(sys.argv) >= 4 else 0.1
        clean_and_output(path, outpath, valid_path, valid_percent, perform_split)