import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
matplotlib.use('agg')
from evaluation.loader import ResultLoader



"""Helper script to generate a a plot showing which predictions occured most often per model"""

def format_label(s):
    # Split the string at 'b'
    parts = s.split('b')
    
    # Extract billions and millions parts
    billions = parts[0]
    millions = parts[1] if len(parts) > 1 else None
    
    # Format the string based on the provided conditions
    if billions == '0':
        return f"{millions}M"
    elif millions:
        return f"{billions}.{millions}B"
    else:
        return f"{billions}B"

def parse_options() -> tuple:
    parser = argparse.ArgumentParser(description="Run machine learning models with different configurations and options.")
    parser.add_argument("-k", "--key", help="Name of the results key", type=str)
    parser.add_argument("-m", "--model", help="Name of the model that should be plotted", type=str)
    parser.add_argument("-c", "--config", help="the configuration used for plotting (original, paraphrased). has to be present in given data")
    args = parser.parse_args()
    return args.key, args.model, args.config


def predictions_analyzer(key, model = None, config = "original"):
    """creates histogram for all predictions, ordered by number of occurences"""
    
    loader = ResultLoader()
    if model is not None:
        results = loader.load(key, model)
        model_name, size = model.split("-")
        data = results[model_name][size][config]["train"]
        df = pd.DataFrame(data)
        predictions = df[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4']].melt()['value']
        # Plot using Seaborn's countplot
        plt.figure(figsize=(10, 6))
        sns.countplot(y=predictions, order=predictions.value_counts().iloc[:10].index, palette="viridis")
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Predictions')        
        plt.grid(axis='x')
        plt.savefig(f"evaluation/plotting/plots/ablations/predictions-analyzer-histogram-{model}-{key}.png", bbox_inches='tight')
    else:
        results = loader.load(key)
        for model_name, model_results in results.items():
            for size, size_results in model_results.items():
                data = size_results[config]["train"]
                df = pd.DataFrame(data)
                predictions = df[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4']].melt()['value']
                # Plot using Seaborn's countplot
                plt.figure(figsize=(10, 6))
                sns.countplot(y=predictions, order=predictions.value_counts().iloc[:10].index, palette="viridis")
                plt.xlabel('Number of Occurrences')
                plt.ylabel('Predictions')
                # replace B character in string with .
                label = format_label(size)
                plt.title(f"Top 10 Predictions for {model_name} {label}")
                plt.grid(axis='x')
                plt.savefig(f"evaluation/plotting/plots/ablations/predictions-analyzer-histogram-{model_name}-{size}-{key}.png", bbox_inches='tight')


def main():
    key, model, config = parse_options()
    if key is None:
        key = "run-all-top-5"
    predictions_analyzer(key=key, model=model, config=config)


if __name__ == "__main__":
    main()