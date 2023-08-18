import csv
import sys
import json
from tqdm import tqdm


"""helper script to convert tsv files to jsonl"""

# allow reading of large rows
csv.field_size_limit(sys.maxsize)

# Open the TSV file and read it into a list of dictionaries
with open('./news_articles/swissdox_newsarticles.tsv', 'r') as tsvfile:
    print("Reading TSV file...")
    reader = csv.DictReader(tsvfile, delimiter='\t')
    rows = list(reader)

# save json docs to jsonl file
with open('news_articles.jsonl', 'w') as jsonlfile:
    print("Writing JSONL file...")
    for doc in tqdm(rows, total=len(rows)):
        json.dump(doc, jsonlfile)
        jsonlfile.write('\n')
