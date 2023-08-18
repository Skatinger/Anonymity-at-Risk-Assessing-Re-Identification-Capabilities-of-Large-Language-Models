import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import Dataset
import matplotlib.lines as mlines
import logging
matplotlib.use('agg')

from evaluation.loader import ResultLoader
from evaluation.single_prediction_eval import SinglePredictionEvaluator
from evaluation.top_k_prediction_eval import TopKPredictionEvaluator

class Plotter():

    def __init__(self, key: str = "") -> None:
        self.key = key

    @staticmethod
    def plotters() -> dict:
        return {
            "accuracy-overview": AccuracyOverviewPlotter(),
            "levenshtein-distance": LevenstheinDistancePlotter(),
            "input-length-ablation": InputLengthAblationPlotter(),
            "page-views-proxy": PageViewsProxyPlotter(),
            "page-edits-proxy": PageEditsProxyPlotter(),
        }

    @staticmethod
    def parse_options() -> tuple:
        parser = argparse.ArgumentParser(description="Run machine learning models with different configurations and options.")
        parser.add_argument("-n", "--name", help="Name of specific chart that should be created", type=str)
        parser.add_argument("-k", "--key", help="Name of the results key", type=str)
        # parser.add_argument("-e", "--exclude", help="Exclude specific models from the run. Format: model_name1,model_name2", type=str)
        parser.add_argument("-m", "--model", help="Name of the model that should be plotted", type=str)
        args = parser.parse_args()
        return args.name, args.key, args.model

    def plot(self, data: dict, gt: Dataset, key: str = "", name = None) -> None:
        self.key = key
        if name is None:
            self.plot_all(data, key, gt)
        else:
            self.plot_one(data, name, key, gt)
    
    def plot_all(self, data, key, gt):
        for name in self.plotters().keys():
            self.plot_one(data, name, key, gt)
            matplotlib.pyplot.close()

    def plot_one(self, data, name, key, gt):
        self.plotters()[name].build(data, key, gt)


class LevenstheinDistancePlotter(Plotter):

    """creates a plot for every model, showing how correctness and edit-distance correlate
    """

    def build(self, data: dict, key: str, gt: Dataset) -> None:
        """takes a dictionary of results and plots a barplot showing the correlation between
            correctness and edit-distance for every model and configuration in the dictionary"""
        for model_class, models in data.items():
            for model, data in models.items():
                configs = (data.keys() & ["original", "paraphrased"])
                for config in configs:
                    # key might be present but without data if metrics were not computed
                    # for that config. in that case, just skip it
                    if not "result" in data[config].keys():
                        continue
                    results = data[config]['result']['data']
                    self._plot_single(results, f"{model_class}-{model}", config, key)

    def _plot_single(self, results: dict, model_name: str, config: str, key: str) -> None:
        """takes a dictionary of results and plots a barplot showing the correlation between"""
        plt.figure(figsize=(12,8))
        font_size = 12
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame({ 'distance': results['distance'], 'correct': results['correct']})
        # round the distance to the closest .1 decimal
        df['distance'] = (df['distance'] / 0.2).round() * 0.2
        df['distance'] = df['distance'].round(1)
        # # Group the data by score and correctness to count the entries with the same score
        df_agg = df.groupby(['distance', 'correct']).size().reset_index(name='count')

        # Map the values of the 'correct' column to their respective terms
        df_agg['correctness'] = df_agg['correct'].replace({1: 'correct', 0: 'incorrect'})
        df_agg.drop('correct', axis=1, inplace=True)

        # Filter the data to only include correct results
        df_agg = df_agg[df_agg['correctness'] == 'correct']

        # Define the barplot using seaborn
        ax = sns.barplot(x='distance', y='count', hue='correctness', data=df_agg)

        # Remove the legend
        ax.get_legend().remove()

        # # Add labels and a title
        plt.xlabel('Levenshtein edit distance', fontsize=font_size)
        plt.ylabel('number of pages', fontsize=font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xticks(rotation=60)
        # set the y axis maximum
        plt.ylim(0, 1200)
        plt.title(f"{model_name}", fontsize=font_size + 8)
        path = f"evaluation/plotting/plots/{key}_{model_name}_{config}_levensthein_distance.png"
        logging.info(f"saving to {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')


class LevenstheinDistanceComparisonPlotter(Plotter):

    """creates a plot with all models, showing how correctness and edit-distance correlate
    """

    def build(self, data: dict, key: str, gt: Dataset) -> None:
        """takes a dictionary of results and plots a barplot showing the correlation between
            correctness and edit-distance for every model and configuration in the dictionary"""
        for model_class, models in data.items():
            for model, data in models.items():
                configs = (data.keys() & ["original", "paraphrased"])
                for config in configs:
                    # key might be present but without data if metrics were not computed
                    # for that config. in that case, just skip it
                    if not "result" in data[config].keys():
                        continue
                    results = data[config]['result']['data']
                    self._plot_single(results, f"{model_class}-{model}", config, key)

    def _plot_single(self, results: dict, model_name: str, config: str, key: str) -> None:
        """takes a dictionary of results and plots a barplot showing the correlation between"""
        plt.figure(figsize=(12,8))
        font_size = 12
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame({ 'distance': results['distance'], 'correct': results['correct']})
        # round the distance to the closest .1 decimal
        df['distance'] = (df['distance'] / 0.2).round() * 0.2
        df['distance'] = df['distance'].round(1)
        # # Group the data by score and correctness to count the entries with the same score
        df_agg = df.groupby(['distance', 'correct']).size().reset_index(name='count')

        # Map the values of the 'correct' column to their respective terms
        df_agg['correctness'] = df_agg['correct'].replace({1: 'correct', 0: 'incorrect'})
        df_agg.drop('correct', axis=1, inplace=True)

        # Filter the data to only include correct results
        df_agg = df_agg[df_agg['correctness'] == 'correct']

        # Define the barplot using seaborn
        ax = sns.barplot(x='distance', y='count', hue='correctness', data=df_agg)

        # Remove the legend
        ax.get_legend().remove()

        # # Add labels and a title
        plt.xlabel('Levenshtein edit distance', fontsize=font_size)
        plt.ylabel('number of pages', fontsize=font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xticks(rotation=60)
        # set the y axis maximum
        plt.ylim(0, 1200)
        plt.title(f"{model_name}", fontsize=font_size + 8)
        path = f"evaluation/plotting/plots/insights/{key}_{model_name}_{config}_levensthein_distance.png"
        logging.info(f"saving to {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')

class AccuracyOverviewPlotter(Plotter):

    """creates a plot including every models accuracies for all configs"""

    def build(self, data, key, gt):
        sizes = []
        accuracies = []
        labels = []

        for model, configs in data.items():
            for config, details in configs.items():
                for dataset, results in details.items():
                    if isinstance(results, dict):  # check if result is a dictionary
                        size = details['size']
                        accuracy = results['result']['accuracy']
                        
                        sizes.append(size)
                        accuracies.append(accuracy)
                        labels.append(f"{model}_{config}_{dataset}")

        plt.figure(figsize=(30, 18))
        plt.scatter(sizes, accuracies)

        for i, label in enumerate(labels):
            plt.annotate(label, (sizes[i], accuracies[i]))

        plt.xlabel('Model Size')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Model Size and Configuration')
        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_{key}.png")
        # ensure pyplot does not run out of memory when too many plots are created
        matplotlib.pyplot.close()

class PageXProxyPlotter(Plotter):
    """creates a plot showing the scores for different page views/edits of wiki pages"""
    def build(self, data, key, gt, proxy, views_df, bin_size=200, cutoff=None):
        gt_df = pd.DataFrame(gt)
        gt_df = gt_df[['id', 'title']]

        fontsize = 26

        # plt.figure(figsize=(10,6))
        
        from matplotlib.colors import ListedColormap
        import matplotlib.ticker as ticker

        if cutoff is not None:
            views_df = views_df[views_df[proxy] < cutoff]

        # Define a color palette with as many distinct colors as there are models
        colors = sns.color_palette('husl', n_colors=len(data.keys()))

        # Create a colormap that maps model keys to colors
        cmap = ListedColormap(colors)

        # Create a figure with two subplots with shared x-axis
        fig, axs = plt.subplots(2, 1, figsize=(20, 16), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

        # Access the first subplot for the first plot
        ax = axs[0] 

        # Keep track of the current color index
        color_index = 0

        # create a list to hold the legend lines
        legend_lines = []

        for model_key in data.keys():
            for size in data[model_key].keys():
                prediction_results = data[model_key][size]['paraphrased']['result']['data']
                predictions_df = pd.DataFrame(prediction_results)

                # First merge the predictions with the ground truth to receive the id of the pages
                df = pd.merge(predictions_df, gt_df, on='title', how='inner')

                # Then join the page views with the predictions
                # Match the ids type to the page_id type
                df['id'] = df['id'].astype(int)
                views_df['page_id'] = views_df['page_id'].astype(int)

                df = pd.merge(df, views_df, left_on='id', right_on='page_id', how='inner')

                df['length_group'] = pd.cut(df[proxy], bins=bin_size)

                # Group by 'length_group' and compute the aggregated values
                grouped = df.groupby('length_group').agg(
                    size=pd.NamedAgg(column=proxy, aggfunc='mean'),
                    accuracy=pd.NamedAgg(column='correct', aggfunc='mean'),
                    last_name_accuracy=pd.NamedAgg(column='last_name_correct', aggfunc='mean')
                ).reset_index()

                # Compute the combined accuracy with given weights
                grouped['weighted_score'] = 0.35 * grouped['accuracy'] + 0.65 * grouped['last_name_accuracy']

                # Use the same color for the regression line, dont show a legend
                plot = sns.regplot(ax=ax, data=grouped, x='size', y='weighted_score', scatter=True, lowess=False,
                                   color=colors[color_index], scatter_kws={'alpha': 0.5, 's': 150}, label=None,
                                   line_kws={'linewidth': 4})
                plot.set_ylabel('PNMS', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plot.set_xlabel("")
                # Create a custom line to add to the legend
                legend_line = mlines.Line2D([], [], color=colors[color_index], label=f'{model_key}-{size}', linewidth=10)
                legend_lines.append(legend_line)
                # place legend below plot

                ax.xaxis.set_major_formatter(ticker.EngFormatter())

                # Move to the next color
                color_index += 1

        plot.legend(handles=legend_lines, loc='upper center', bbox_to_anchor=(0.5, + 1.1), ncol=5, fontsize=fontsize)
        # Access the second subplot for the second plot

        # Adjust tick label size for the first subplot
        axs[0].tick_params(axis='both', which='major', labelsize=fontsize)
        # Adjust tick label size for the second subplot
        axs[1].tick_params(axis='both', which='major', labelsize=fontsize)


        ax = axs[1]
        # lower plot
        plot = sns.histplot(ax=ax, data=views_df, x=proxy, bins=120, log_scale=(False, True), color='skyblue', edgecolor='black')

        if proxy == 'page_views':
            ax.set_xlabel('Number of Page Views', fontsize=fontsize)
        else:
            ax.set_xlabel('Number of Page Edits', fontsize=fontsize)
        ax.set_ylabel('Number of Pages', fontsize=fontsize)
        ax.xaxis.set_major_formatter(ticker.EngFormatter())

        plt.tight_layout()

        # save to file
        plt.savefig(f"evaluation/plotting/plots/proxies/combined_plot_{proxy}.png", dpi=300, bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        matplotlib.pyplot.close()

class PageViewsProxyPlotter(PageXProxyPlotter):
    """creates a plot showing the scores for different page views of wiki pages"""
    def build(self, data, key, gt):
        views_df = pd.read_csv(f"dataset/wiki_page_views.csv")
        # for the page views data, entries above 1000000 are outliers, so we remove them
        cutoff = 1000000
        binsize = 25
        return super().build(data, key, gt, 'page_views', views_df, binsize, cutoff)

class PageEditsProxyPlotter(PageXProxyPlotter):
    """creates a plot showing the scores for different page edits of wiki pages"""
    def build(self, data, key, gt, bin_size=150):
        views_df = pd.read_csv(f"dataset/wiki_page_edits.csv")
        # remove any entries with None values
        views_df = views_df[views_df['page_edits'].notna()]
        # for the page edits data, entries above 100000 are outliers, so we remove them
        cutoff = 200
        bin_size = 30
        return super().build(data, key, gt, 'page_edits', views_df, bin_size, cutoff)

class InputLengthAblationPlotter(Plotter):
    """creates a plot showing the scores for different lengths of wiki pages
       expectes a single model to be passed in the data used for plotting"""

    def build(self, data, key, gt):
        gt_df = pd.DataFrame(gt)
        gt_df = gt_df[['title', 'text']]

        plt.figure(figsize=(10,6))
        
        from matplotlib.colors import ListedColormap

        # Define a color palette with as many distinct colors as there are models
        colors = sns.color_palette('husl', n_colors=len(data.keys()))

        # Create a colormap that maps model keys to colors
        cmap = ListedColormap(colors)

        # Keep track of the current color index
        color_index = 0

        # create a list to hold the legend lines
        legend_lines = []

        for model_key in data.keys():
            for size in data[model_key].keys():
                prediction_results = data[model_key][size]['paraphrased']['result']['data']
                predictions_df = pd.DataFrame(prediction_results)

                df = pd.merge(predictions_df, gt_df, on='title', how='inner')

                df['text_length'] = df['text'].apply(len)
                # convert text length to words
                df['text_length'] = df['text_length'] / 5
                df['length_group'] = pd.cut(df['text_length'], bins=50)

                grouped = df.groupby('length_group').agg(
                    size=pd.NamedAgg(column='text_length', aggfunc='mean'),
                    accuracy=pd.NamedAgg(column='correct', aggfunc='mean')
                ).reset_index()

                # Use the same color for the regression line
                sns.regplot(data=grouped, x='size', y='accuracy', scatter=False, lowess=True, color=colors[color_index])
                
                # create a custom line to add to the legend
                legend_line = mlines.Line2D([], [], color=colors[color_index], label=f'{model_key}-{size}')
                legend_lines.append(legend_line)
                
                # Move to the next color
                color_index += 1

        plt.xlabel('average text length [words]')
        plt.ylabel('partial name match score')
        plt.ylim(0, 1)
        plt.title(f"partial name match score by text length [words]\n {key}")

        # Add legend manually
        plt.legend(handles=legend_lines, title='model')
        
        ###### saving, dont change below here #####
        plt.savefig(f"evaluation/plotting/plots/proxies/plot_input_length_{key}.png")
        # ensure pyplot does not run out of memory when too many plots are created
        matplotlib.pyplot.close()


def main():
    # key for result dataset from command line arguments
    name, key, model = Plotter.parse_options()
    assert key is not None, "No key provided"
    print("creating loader")
    loader = ResultLoader()
    print("loading ground truth")
    gt = loader.load_gt()
    if model is not None:
        results = loader.load(key, model)
    else:
        results = loader.load(key)


    configs = ['paraphrased']
    logging.info(f"Using configs {configs}. If you want to change this, adapt variable configs in plotter.py.")

    computed = TopKPredictionEvaluator.compute_metrics(gt, results, configs)
    # import pdb; pdb.set_trace()
    plotter = Plotter()
    plotter.plot(name=name, gt=gt, data=computed, key=key)


if __name__ == "__main__":
    main()