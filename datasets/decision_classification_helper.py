#!/usr/bin/python
import os
import sys
import pandas as pd

doc = """This script helps to quickly judge if a news article is interesting for reidentification.
         Helpful usually means it contains an identified person (e.g. Max Muster) and an indication
         to which court decisions it might belong.

         Usage:
            python3 judge.py <articles.tsv>

            The input file is expected to have the tab separeted columns:
            - id
            - pubtime
            - medium_code
            - medium_name
            - rubric
            - regional
            - doctype
            - doctype_description
            - language
            - char_count
            - dateline
            - head
            - subhead
            - content_id
            - content
            No header row is required.

        - For each datapoint entered, the script will ask you to judge if it is interesting or not.
        - The interesting articles will be written to a file called 'interesting.tsv' in the same
          directory as the input file.
        - Decisions which have been judged already (present in 'interesting.tsv' will be skipped.
"""


def help():
    print(doc)


def columns():
    return [
        'id', 'pubtime', 'medium_code', 'medium_name', 'rubric', 'regional',
        'doctype', 'doctype_description', 'language', 'char_count', 'dateline',
        'head', 'subhead', 'content_id', 'content'
    ]


# helper to print a news article text with interesting words highlighted
def print_content(text):
    # replace html tags
    text = text.replace('<tx>', '\n')
    text = text.replace('</tx>', '\n')
    text = text.replace('<ld>', '\n\033[95m')
    text = text.replace('</ld>', '\033[0m\n')
    text = text.replace('<p>', '\n')
    text = text.replace('</p>', '\n')
    text = text.replace('<zt>', '\n\n\033[1m')
    text = text.replace('</zt>', '\033[0m\n')
    text = text.replace('<au>', '')
    text = text.replace('</au>', '')
    print(text)


def print_decision(row):
    print("============= ID: {} ===============================================".format(row['id']))
    print("Pubtime: {}".format(row['pubtime']))
    print("Medium: {} ({})".format(row['medium_name'], row['medium_code']))
    print("Rubric: {}".format(row['rubric']))
    print("Regional: {}".format(row['regional']))
    print("Doctype: {} ({})".format(row['doctype_description'], row['doctype']))
    print("Language: {}".format(row['language']))
    print("Char count: {}".format(row['char_count']))
    print("Dateline: {}".format(row['dateline']))
    print("Head: {}".format(row['head']))
    print("Subhead: {}".format(row['subhead']))
    print_content(row['content'])
    print("======================================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing input file argument.")
        help()
        exit(1)

    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        help()
        exit(0)

    input_file = sys.argv[1]
    output_file = os.path.join(os.path.dirname(input_file), "processed.tsv")

    # read in the interesting articles
    if not os.path.exists(output_file):
        print("No output file found, creating new file.")
        with open(output_file, "w") as f:
            df = pd.DataFrame(columns=[columns() + ['interesting']])
            df.to_csv(f, sep='\t', index=False)

    processed = pd.read_csv(output_file, sep='\t')

    # read in the input file
    print("Reading input file {}. (This might take a while)".format(input_file))
    input_df = pd.read_csv(input_file, sep='\t', header=None)
    # set the header, as files usually dont have a header
    input_df.columns = columns()

    # filter out the articles which have already been judged
    print("Filtering out already judged articles.")
    input_df = input_df[~input_df['id'].isin(processed['id'])]

    assert len(input_df) > 0, "No articles left to judge."

    print("Starting article judgment. Press q to quit.\n\n")
    # list tracking the decisions which have been judged interesting, will be appended to the interesting articles
    # dataframe at the end, as performance is bad if rows are appended one by one
    to_append = []
    for index, decision in input_df.iterrows():
        print_decision(decision)
        print("Is this decision interesting? (q for quit)\ny/n/q:")
        response = input()
        if str(response) == 'y':
            # append decision to interesting.csv
            to_append.append(decision)
        elif str(response) == 'q':
            break
        elif str(response) == 'n':
            reason = input("Reason? (defaults to missing person)\nOptions:\n - m missing person\n - <custom reason>\n\nm/q/custom reason:")
            if str(reason) == 'q':
                break
            elif (str(reason) == 'm' or str(reason) == ''):
                reason = 'missing person'
            decision['interesting'] = reason
            to_append.append(decision)
        else:
            print("Invalid response. Skipping decision.\n\n")

    # append the decisions which have been judged interesting to the interesting articles dataframe
    processed = pd.concat([processed, pd.DataFrame(to_append)])
    print("Writing interesting articles to {}".format(output_file))
    # write interesting articles to file
    processed.to_csv(output_file, sep='\t', index=False)
