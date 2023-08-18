import sys
import pandas as pd
import numpy as np
from evaluation.loader import ResultLoader
from datasets import load_dataset
from names_dataset import NameDataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from matplotlib.ticker import FuncFormatter

name_dataset = NameDataset()

global classification
classification = {
        "exists in text": {},
        "not a name": {},
        "no prediction": {},
        "just a letter": {},
        "good": {},
    }


# Define the function to format tick values
def kilo_formatter(x, pos):
    if x == 0:
        return '0'
    elif x < 1000:
        return f'{x/1000:.1f}k'
    else:
        return f'{x//1000}k'

def is_name(word: str) -> bool:
    """checks if a word is a name by checking if it is in the names dataset"""
    return name_dataset.search(word)["first_name"] is not None or name_dataset.search(word)["last_name"] is not None

def join_predictions(row, name):

    # dict to track which values are what category (present in text, not a name, good)
    row = row.to_dict()
    string = ""
    for key, value in row.items():
        if value is None:
            classification["no prediction"][name] += 1
        elif key.startswith("prediction") and value not in row["masked_text_original"]:
            if is_name(value):
                string += f" {value}"
                classification["good"][name] += 1
            else:
                # 2 letters or less without whitespace are not names
                if len(value.replace(" ", "")) < 3:
                    classification["just a letter"][name] += 1
                else:
                    classification["not a name"][name] += 1
        else:
            # value is already in masked_text_original
            classification["exists in text"][name] += 1
    return string

def main():
    assert len(sys.argv) > 1, "Please provide a name of results folder as an argument"

    dataset_key = sys.argv[1]

    loader = ResultLoader()
    print("loading ground truth")
    gt = loader.load_rulings_gt()
    gt_df = gt.to_pandas()

    print("loading rulings dataset")
    rulings = load_dataset("rcds/swiss_rulings", split="train")
    # only keep rulings with decision_id in gt ids
    gt_ids = gt["id"]
    rulings = rulings.filter(lambda x: x['decision_id'] in gt_ids)

    print("loading predictions")
    predictions = loader.load(dataset_key)
    
    full_rulings_df = rulings.to_pandas()
    # Rename the column "decision_id" to "page_id" in full_rulings_df to match with other dataframes
    full_rulings_df = full_rulings_df.rename(columns={"decision_id": "page_id"})

    if full_rulings_df['page_id'].duplicated().any():
        print("Duplicates found in 'page_id' of 'full_rulings_df'")


    model_columns = []
    for model_class_name, model_class in predictions.items():
        for model_size, model in model_class.items():
            print(f"processing model {model_class_name}-{model_size}")
            # Convert the predictions dataset to a pandas dataframe
            predictions_df = model['original']['train'].to_pandas()

            # join the ground truth onto the predictions
            predictions_df = pd.merge(predictions_df, gt_df, left_on="page_id", right_on="id")

            # swiss_bert accidentally predicted some rows twice due some special handling.
            # remove the duplicate rows
            if predictions_df['page_id'].duplicated().any():
                predictions_df = predictions_df.drop_duplicates(subset=['page_id'])

            # prepare classification dict
            for key in classification.keys():
                classification[key][f"{model_class_name}-{model_size}"] = 0

            # add a column with the model name and size and fill it with the joined predictions
            model_name = f"{model_class_name}-{model_size}"
            predictions_df[f"{model_class_name}-{model_size}"] = predictions_df.apply(join_predictions, name=model_name, axis=1)

            # join the predictions dataframe to the rulings dataframe on "page_id"
            full_rulings_df = pd.merge(full_rulings_df, predictions_df[["page_id", f"{model_class_name}-{model_size}"]], on="page_id")

            # add the model name and size to the list of columns
            model_columns.append(f"{model_class_name}-{model_size}")

    
    # only keep the columns we need for the final dataframe
    final_df = full_rulings_df[["page_id", "file_id", "file_name", "file_number"] + model_columns]

    # Create a new column 'total_length' that contains the sum of the character lengths of the last columns
    final_df['total_length'] = final_df[model_columns].apply(lambda row: sum(row.fillna('').apply(len)), axis=1)

    # Sort the DataFrame based on the 'total_length' column
    final_df = final_df.sort_values(by='total_length', ascending=False)

    # Delete the 'total_length' column as it's no longer needed
    del final_df['total_length']

    # Save the final dataframe to a CSV file
    final_df.to_csv(f"{dataset_key}-for-BG.csv", index=False)

    # plot a histogram of the lengths of classification categories
    labels = list(classification.keys())
    models = list(classification[labels[0]].keys())

    # Calculate total scores for each model and sort models accordingly
    total_scores = {model: sum(classification[label][model] for label in labels) for model in models}
    models.sort(key=lambda x: total_scores[x], reverse=True)

    x = np.arange(len(labels))  # the label locations
    width = 0.17  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 5))
    rects = []
    for i, model in enumerate(models):
        model_values = [classification[label][model] for label in labels]
        rects.append(ax.bar(x - width*len(models)/2 + i*width, model_values, width, label=model))

    # Drawing vertical lines to separate the groups
    for i in range(len(labels) - 1):
        line_position = x[i] + width*len(models)/2
        ax.axvline(line_position, color='gray', linestyle='--', linewidth=1.2)
    
    # Adjust the x-axis limits to reduce whitespace
    # ax.set_xlim(left=x[0] - width*len(models)/2 + 0.2)  # Adjust the 0.5 value as needed

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_title('Type of prediction by model', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    plt.xticks(rotation=25, ha='right')
    # Set the custom formatter for the y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(kilo_formatter))
    # sent font size of y axis ticks to 20
    ax.tick_params(axis='y', labelsize=20)
    legend = ax.legend(fontsize=18)
    plt.grid(axis="y")
    # legend.set_title('models', prop={'size': 26})

    # Move x-tick labels to the center of the sections
    for tick in ax.get_xticklabels():
        tick.set_ha('center')

    # plt.xlabel('Classified as', fontsize=20)
    plt.ylabel('number of predictions', fontsize=20)

    plt.savefig(f"evaluation/plotting/plots/legal/{dataset_key}-histogram.png", bbox_inches='tight')


if __name__ == "__main__":
    main()