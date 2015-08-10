import sqlite3
import wikipedia_ner.parse
import gzip, pickle
import random
import argparse
from os.path import join, isdir, exists

def uber_decode(s):
    try:
        return s.decode("unicode_escape")
    except:
        return s.decode("utf-8")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--out',   type =str, required=True)
    parser.add_argument('--index2target',   type =str, required=True)
    parser.add_argument('--total', type=int, default=500)
    parser.add_argument('--max_labels', type=int, default=100)
    parser.add_argument('--tsv', action="store_true", default=False)
    args = parser.parse_args()
    return args

def write_samples(inpath, outpath, index2target, total=500, tsv=False, max_labels=100):
    """
    Write Samples
    -------------

    Output some sample protobuff files in gzipped
    form in the sample folder to check compatibility
    between Python and C++.


    Each protobuff file contains several example
    sentences from Wikipedia with words tokenized
    and anchor tag text lifted and annotated
    with the destination Wikipedia article.

    """
    if not tsv:
        assert(isdir(outpath)), "For protobuff output, out path must be a directory."

    sqlite_conn = sqlite3.connect(
        inpath,
        detect_types=sqlite3.PARSE_DECLTYPES)
    insert_into_db, update_in_db, update_lines_in_db, get_obj_from_db, get_lines_from_db = wikipedia_ner.parse.sqlite_utils.create_schema(sqlite_conn, [
            ("lines", "pickle"),
            ("parents", "pickle")
        ],
        "articles")

    f = gzip.open(index2target, "rb")
    targets = [uber_decode(line).strip() for line in f]
    f.close()

    print("Got all possible labels")

    random.shuffle(targets)

    print("Shuffled labels")

    curr = 0
    corpuses = []
    print("Collecting corpus")
    for k, value in enumerate(targets):
        if curr > total:
            break
        objs = get_lines_from_db(value)
        if objs is not None and type(objs[0]) is not list:
            corpuses.append(objs[0])
            curr += 1
    if tsv:
        outpath_fname = outpath
        if isdir(outpath_fname):
            outpath_fname = join(outpath_fname, "train.tsv.gz")

        with gzip.open(outpath_fname, "wt") as fout:
            for k, corpus in enumerate(corpuses):
                print("Saving corpus %d/%d\r" % (k,total), flush=True, end="")
                for example in corpus.example:
                    if len(example.trigger) <= max_labels:
                        num_non_empty_triggers = 0
                        for trigger in example.trigger:
                            if len(trigger.trigger.strip()) > 0:
                                num_non_empty_triggers += 1

                        if num_non_empty_triggers > 0:
                            fout.write(" ".join(example.words))
                            for trigger in example.trigger:
                                fout.write("\t")
                                fout.write(trigger.trigger)
                            fout.write("\n")
    else:
        for k, corpus in enumerate(corpuses):
            print("Saving corpus %d/%d\r" % (k,total), flush=True, end="")
            with gzip.open(join(outpath, "%d.gz" % (k,)), "wb") as f:
                f.write(corpus.SerializeToString())

if __name__ == "__main__":
    args = parse_args()
    write_samples(args.input, args.out, args.index2target, total=args.total, tsv=args.tsv, max_labels=args.max_labels)
