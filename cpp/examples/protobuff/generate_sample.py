import sqlite3
import wikipedia_ner.parse
import gzip, pickle
def write_samples():
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
    sqlite_conn = sqlite3.connect(
        "/Users/jonathanraiman/Desktop/crawl_results/crawl_results/dump.db",
        detect_types=sqlite3.PARSE_DECLTYPES)
    insert_into_db, update_in_db, update_lines_in_db, get_obj_from_db, get_lines_from_db = wikipedia_ner.parse.sqlite_utils.create_schema(sqlite_conn, [
            ("lines", "pickle"),
            ("parents", "pickle")
        ],
        "articles")

    f = gzip.open("/Users/jonathanraiman/Desktop/crawl_results/targets.gz")
    targets = pickle.load(f)
    f.close()

    max_acc = 50
    curr = 0
    corpuses = []
    for k, value in enumerate(targets.values()):
        if curr > max_acc:
            break
        objs = get_lines_from_db(value)
        if objs is not None and type(objs[0]) is not list:
            corpuses.append(objs[0])
            curr += 1

    for k, corpus in enumerate(corpuses):
        with gzip.open("sample/%d.gz" % (k), "wb") as f:
            f.write(corpus.SerializeToString())

if __name__ == "__main__":
    write_samples()