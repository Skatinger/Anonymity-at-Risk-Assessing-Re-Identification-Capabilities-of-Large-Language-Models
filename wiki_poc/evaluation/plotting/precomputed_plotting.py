# plotting class which uses precomputed results

import sys
from evaluation.loader import ResultLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import matplotlib
import math
from adjustText import adjust_text
from collections import defaultdict


matplotlib.use('agg')


"""
Plotting class that can play all kinds of different plots. Plotting methods assume that wikipedia results are used,
some expect specific models to be present in the data, or to contain a specific format.
All "precomputed" results are loaded from wiki_poc/evaluation/results. You can pass the name of the results json
file (except the -results.json part) as second argument to run on your results.

Uncomment the desired plotting functions on line 50 for your specific result.
Plots will be saved to the corresponding folder under wiki_poc/evaluation/plotting/plots/

Note: This class is built as a one-shot tinkering playground and therefore not very stable.
"""

class PrecomputedPlotting():

    def __init__(self, key, model_name = None):
        self.size_label = "Size in Billion Parameters"
        self.accuracy_label = "Accuracy"
        print("loading results...")
        self.results = ResultLoader().load_computed(key, model_name)
        self.set_baseline_values()

    
    def set_baseline_values(self):
        logging.info("setting baseline values, using: ")
        # hardcoded baselines for weighted scores
        self.baseline_random = 0.03
        self.baseline_majority = 0.11
        logging.info(f"random_full_name: {self.baseline_random}")
        logging.info(f"majority_full_name: {self.baseline_majority}")

    def plot(self):
        # convert results to dataframe format for easier plotting
        prepared_df = self.convert_to_df(self.results)
        # self.plot_normal_to_instructional(self.results, prepared_df)
        # self.plot_normal_to_instructional_barplot(self.results, prepared_df)
        self.plot_accuracy_progression(self.results, prepared_df)
        # self.plot_best_performers(self.results, prepared_df)
        # self.plot_with_huge(self.results, prepared_df)
        # self.plot_accuracy_overview(self.results)
        # self.plot_accuracy_overview_with_legend(self.results, prepared_df)
        # self.plot_accuracy_input_size_comparison(self.results, prepared_df)
        # self.input_length_progression(self.results, prepared_df)
        # self.sampling_method_comparison(self.results, prepared_df)
        # self.plot_accuracy_overview_with_legend_and_size(self.results, prepared_df)
        # self.plot_model_types_comparison(self.results, prepared_df)
        # self.plot_model_types_comparison_scatter(self.results, prepared_df)
        # self.compute_difference_paraphrased_to_original(self.results, prepared_df)
        # self.tabulate_results_to_latex(self.results)


    def compute_difference_paraphrased_to_original(self, results, prepared_df):
        # Define metrics to be calculated
        metrics = ["accuracy", "precision", "last_name_accuracy", "last_name_precision", "weighted_score"]

        # Initialize counters and sums for original and paraphrased configurations
        counters = defaultdict(int)
        sums = defaultdict(lambda: defaultdict(float))

        # Iterate through the data and calculate the sums and counts
        for model, model_data in results.items():
            if model == "key":
                continue
            for config in ["original", "paraphrased"]:
                if config not in model_data.keys():
                    continue
                counters[config] += 1
                for metric in metrics:
                    sums[config][metric] += model_data[config][metric]

        # Calculate averages
        averages = defaultdict(lambda: defaultdict(float))
        for config in ["original", "paraphrased"]:
            for metric in metrics:
                averages[config][metric] = sums[config][metric] / counters[config] if counters[config] else 0

        # Create a table with the results
        table_data = [["Configuration"] + metrics]
        for config in ["original", "paraphrased"]:
            table_data.append([config] + [averages[config][metric] for metric in metrics])
        
        # convert table_data to dataframe
        table_data = pd.DataFrame(table_data[1:], columns=table_data[0])

        # Initialize variance and standard deviation dictionaries
        variances = defaultdict(lambda: defaultdict(float))
        standard_deviations = defaultdict(lambda: defaultdict(float))

        # Calculate variances
        for model, model_data in results.items():
            if model == "key":
                continue
            for config in ["original", "paraphrased"]:
                if config not in model_data.keys():
                    continue
                for metric in metrics:
                    variances[config][metric] += (model_data[config][metric] - averages[config][metric])**2

        # Calculate standard deviations
        for config in ["original", "paraphrased"]:
            for metric in metrics:
                variances[config][metric] = variances[config][metric] / counters[config] if counters[config] else 0
                standard_deviations[config][metric] = math.sqrt(variances[config][metric])

        # Calculate standard deviation of the mean
        standard_errors = defaultdict(lambda: defaultdict(float))
        for config in ["original", "paraphrased"]:
            for metric in metrics:
                standard_errors[config][metric] = standard_deviations[config][metric] / math.sqrt(counters[config]) if counters[config] else 0

        # Create rows for standard errors
        original_errors = ['original_std_error'] + [standard_errors['original'][metric] for metric in metrics]
        paraphrased_errors = ['paraphrased_std_error'] + [standard_errors['paraphrased'][metric] for metric in metrics]

        original_errors_df = pd.DataFrame([original_errors], columns=table_data.columns)
        paraphrased_errors_df = pd.DataFrame([paraphrased_errors], columns=table_data.columns)

        table_data = pd.concat([table_data, original_errors_df, paraphrased_errors_df], ignore_index=True)

        # save result to file
        latext_text = table_data.to_latex(index=False, float_format="{:0.2f}".format)
        with open(f"evaluation/plotting/plots/text_results/paraphrased_to_original_{results['key']}.tex", "w") as f:
            f.write(latext_text)

    @staticmethod
    def sampling_method_comparison(results, _df):        
        data_for_df = {}

        # replace names with correct names
        # fix the names
        names = {
            "bloomz": "BLOOMZ",
            "flan_t5": "Flan_T5",
            "roberta": "RoBERTa",
            "t5": "T5",
            "mt0": "mT0",
            "bloom": "BLOOM",
            "cerebras": "Cerebras-GPT",
            "pythia": "Pythia",
            "t0": "T0",
            "incite_instruct": "INCITE-Instruct",
        }

        def format_label(s):
            # Split the string at 'b'
            parts = s.split('B')
            
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

        for key, value in results.items():
            if key != 'key':
                method = key.split('--')[-1]
                model = key.split('--')[0]
                # and now replace the model name with the correct name
                model_name, model_size = model.split('-')
                # import pdb; pdb.set_trace()
                # convert size to number
                model_size = format_label(model_size)
                model = names[model_name] + " " + model_size
                if method == "beam_search_sampling":
                    method = "beam-search"
                elif method == "sampling":
                    method = "top-k\nsampling"
                elif method == "beam_search":
                    method = "beam search\n(n_beams=5)"
                elif method == "top_p_sampling":
                    method = "top-p\n(p=0.92)"
                elif method == "top_k_sampling":
                    method = "top-k\n(k=50)"
                elif method == "nucleus_sampling":
                    method = "nucleus\n(k=50, p=0.92)"
                elif method == "random_sampling":
                    method = "random"
                elif method == "greedy":
                    method = "greedy\n(top-1)"
                elif method == "top_k_sampling_kruns":
                    method = "top-k\n(k=5)"
                # accuracy = value['paraphrased']['accuracy']
                weighted_score = value['paraphrased']['weighted_score']
                # data_for_df.append([method, accuracy])
                if data_for_df.get(model) is None:
                    data_for_df[model] = [[method, weighted_score]]
                else:
                    data_for_df[model].append([method, weighted_score])
        
        # Restructuring data
        grouped_data = {}
        for model, values in data_for_df.items():
            for method, score in values:
                if method not in grouped_data:
                    grouped_data[method] = {}
                grouped_data[method][model] = score

        methods = list(grouped_data.keys())
        models = list(data_for_df.keys())

        # Average scores for each method across all models
        average_method_scores = {method: np.mean(list(scores.values())) for method, scores in grouped_data.items()}
        sorted_methods = sorted(average_method_scores, key=average_method_scores.get, reverse=True)

        # Average scores for each model across all methods
        average_model_scores = {model: np.mean([grouped_data[method][model] for method in sorted_methods]) for model in models}
        sorted_models = sorted(average_model_scores, key=average_model_scores.get, reverse=True)

        # Plotting the sorted data

        x = np.arange(len(sorted_methods))
        bar_width = 0.25  # Adjust as needed for number of models

        fig, ax = plt.subplots(figsize=(21, 12))

        # For each sorted model, plot bars for each sorted method
        for idx, model in enumerate(sorted_models):
            scores = [grouped_data[method][model] for method in sorted_methods]
            ax.bar(x + idx * bar_width, scores, width=bar_width, label=model, align='center')

        # ax.set_title('Weighted Score by Method and Model (Sorted)')
        ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(sorted_methods)

        # smaller y ticks
        # Determine the y-axis range
        y_min = 0
        y_max = max(max(grouped_data[method][model] for method in sorted_methods) for model in sorted_models) + 0.05  # Adding a small margin for clarity

        # Generate y ticks with smaller steps
        y_ticks = np.arange(y_min, y_max, 0.05)
        ax.set_yticks(y_ticks)


        legend = plt.legend(fontsize=36.5, framealpha=1,
                            loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(models))

        plt.ylabel('W-PNMS', fontsize=38)
        # plt.title('Comparison of Different Generation Methods (top 5)\nincite_instruct-3b')
        plt.xticks(rotation=35, ha='right')
        # increase font size of labels
        plt.tick_params(axis='both', which='major', labelsize=38)
        ax.set_xlabel('')
        plt.grid(axis="y")
        plt.tight_layout()  # adjusts subplot params so that the subplot fits into the figure area

        plt.savefig('evaluation/plotting/plots/ablations/sampling_method_comparison.png', dpi=400, bbox_inches='tight')


    def plot_with_huge(self, results, df2):
        # Extract the class of the model from the 'model' column
        my_font_size = 24

        # Find the row index with the highest accuracy for each model
        idx = df2.groupby('model_class')['accuracy'].idxmax()

        # Select the rows corresponding to the highest accuracy for each model
        best_rows = df2.loc[idx]
        df2 = best_rows

        plt.figure(figsize=(20, 14))
        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=300, markers=True, legend=False)

        # Add labels to each point with adjusted positions
        for i, row in df2.iterrows():
            label_length = len(row['model'])
            mv_left = label_length * 2 + -5
            plt.annotate(row['model_class'], (row['size'], row['accuracy']), xytext=(-mv_left, 12), textcoords='offset points',
                         fontsize=my_font_size)

        # baselines hardcoded (TODO: make dynamic or change if test set changes)
        plt.axhline(y=self.random_baseline, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=self.majority_baseline, color='orange', linewidth=2.5, label="majority names")

        # Annotate the baselines
        plt.annotate("random names", (df2['size'].min(), 0.06), xytext=(-10, 5), textcoords='offset points', color='blue', fontsize=my_font_size)
        plt.annotate("majority names", (df2['size'].min(), 0.13), xytext=(-10, 5), textcoords='offset points', color='orange', fontsize=my_font_size)

        # Set labels and title
        plt.xlabel("size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("partial name match score", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # Set x-axis to logarithmic scale
        plt.xscale('log')

        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_best_performers_with_huge_{results['key']}.png") # , bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()
    
    @staticmethod
    def plot_accuracy_progression(results, df2):
        my_font_size = 42
        plt.figure(figsize=(20, 14))
        # only keep the interesting models, e.g. models which were evaluated in different sizes
        interesting_models = ["bloomz", "flan_t5", "roberta", "t5", "mt0", "bloom", "cerebras", "pythia", "t0"]
                              # , "llama2"] # ditched roberta_squad and llama2, not interesting

        # remove all models that are not in the interesting models list
        df2 = df2[df2['model_class'].isin(interesting_models)]

        # fix the names
        names = {
            "bloomz": "BLOOMZ",
            "flan_t5": "Flan_T5",
            "roberta": "RoBERTa",
            "t5": "T5",
            "mt0": "mT0",
            "bloom": "BLOOM",
            "cerebras": "Cerebras-GPT",
            "pythia": "Pythia",
            "t0": "T0"
        }

        # replace model_class names with name from the names dict
        df2['model_class'] = df2['model_class'].replace(names)

        # remove any models above 20 billion parameters, as they are not interesting because
        # they pull apart the plot too much
        df2 = df2[df2['size'] < 20]

        # group it
        grouped_data = df2.groupby('model_class').apply(lambda x: x.sort_values('size')).reset_index(drop=True)

        # Get unique model groups and their corresponding colors
        unique_groups = grouped_data['model_class'].unique()
        color_palette = sns.color_palette('tab10', n_colors=len(unique_groups))

        # scatter points
        sns.scatterplot(data=grouped_data, x='size', y='weighted_score', hue='model_class', s=500, markers=True, legend=False)

        # Plot separate lines for each group with corresponding colors
        for group, color in zip(unique_groups, color_palette):
            group_data = grouped_data[grouped_data['model_class'] == group]
            plt.plot(group_data['size'], group_data['weighted_score'], color=color, label=group, linewidth=7)

        # Set labels and title
        plt.xlabel("model size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("W-PNMS", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # Add legend, place it outside of plot
        legend = plt.legend(fontsize=my_font_size - 3, framealpha=1)
        # increase linewidth of legend lines
        for line in legend.get_lines():
            line.set_linewidth(8)

        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/ablations/plot_accuracy_progression_{results['key']}.png", bbox_inches='tight', dpi=300)
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()

    @staticmethod
    def input_length_progression(results, df):
        my_font_size = 32
        plt.figure(figsize=(20, 7.5))

        # use input size as hue
        df['input_size'] = df['model_class'].apply(lambda x: x.split("_")[-1])
        # convert input size to integers
        df['input_size'] = df['input_size'].apply(lambda x: int(x))
        # remove input sizes from model class names
        df['model_class'] = df['model_class'].apply(lambda x: "_".join(x.split("_")[:-1]))

        names = {
            "bloomz": "BLOOMZ",
            "flan_t5": "Flan_T5",
            "roberta": "RoBERTa",
            "t5": "T5",
            "mt0": "mT0",
            "bloom": "BLOOM",
            "cerebras": "Cerebras-GPT",
            "pythia": "Pythia",
            "t0": "T0",
            "incite_instruct": "INCITE-Instruct",
            "roberta_squad": "RoBERTa-SQuAD",
        }

        # replace model_class names with name from the names dict
        df['model_class'] = df['model_class'].replace(names)

        # group it
        df = df.sort_values('input_size', ascending=False)
        grouped_data = df.groupby('model_class').apply(lambda x: x.sort_values('input_size')).reset_index(drop=True)

        # Get unique model groups and their corresponding colors
        unique_groups = grouped_data['model_class'].unique()
        color_palette = sns.color_palette('tab10', n_colors=len(unique_groups))

        # scatter points
        sns.scatterplot(data=grouped_data, x='input_size', y='accuracy', hue='model_class', s=450, markers=True, legend=False)

        # Plot separate lines for each group with corresponding colors
        for group, color in zip(unique_groups, color_palette):
            group_data = grouped_data[grouped_data['model_class'] == group]
            # concat model_class and model_size to get the label
            if group_data['size'].iloc[0] > 1:
                num = group_data['size'].iloc[0]
                number = int(num) if num == round(num) else num
                label = f"{group} {number}B"
            else:
                num = group_data['size'].iloc[0]
                number = int(num) if num == round(num, 0) else num
                number = round(number * 1000)
                label = f"{group} {number}M"

            plt.plot(group_data['input_size'], group_data['accuracy'], color=color, label=label, linewidth=9)

        # Set labels and title
        plt.xlabel("input size [characters]", fontsize=my_font_size)
        plt.ylabel("PNMS", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)
        plt.ylim(0, 0.6)
        plt.xlim(400, 4100)

        # Add legend outside on the right of the plot
        legend = plt.legend(fontsize=my_font_size - 4, framealpha=1,
                            loc='upper center', bbox_to_anchor=(0.47, -0.15), ncol=3)
                            # , loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0.)
        # increase linewidth of legend lines
        for line in legend.get_lines():
            line.set_linewidth(9)

        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/ablations/plot_input_length_progression_{results['key']}.png",
                    bbox_inches='tight', dpi=300)
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()
    
    def plot_wiki_edits_proxy(results, df2):
        """Plot the wiki edits proxy metric for each model
           this shows the number of edits to a page compared to the length of the page"""


    @staticmethod
    def plot_best_performers(results, df2):
        # Extract the class of the model from the 'model' column
        my_font_size = 24

        # Find the row index with the highest accuracy for each model
        idx = df2.groupby('model_class')['accuracy'].idxmax()

        # Select the rows corresponding to the highest accuracy for each model
        best_rows = df2.loc[idx]
        df2 = best_rows

        plt.figure(figsize=(20, 14))
        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=300, markers=True, legend=False)

        # Add labels to each point with adjusted positions
        for i, row in df2.iterrows():
            label_length = len(row['model'])
            # if the point is to the right of the plot, move the label to the left
            if row['size'] == df2['size'].max():
                mv_left = - (label_length * 10)
            # small hack for gptj, to make it fit as well
            elif row['model_class'] == "gptj":
                mv_left = - (label_length * 8)
            # otherwise, move it to the right
            else:
                mv_left = 10
            plt.annotate(row['model_class'], (row['size'], row['accuracy']), xytext=(mv_left, -7), textcoords='offset points',
                         fontsize=my_font_size)

        # baselines hardcoded (TODO: make dynamic or change if test set changes)
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("partial name match score", fontsize=my_font_size)

        # scale x as log
        plt.xscale('log')

        # Annotate the baselines
        # plt.annotate("random names", (df2['size'].min(), 0.06), xytext=(-10, 5), textcoords='offset points', color='blue', fontsize=my_font_size)
        # plt.annotate("majority names", (df2['size'].min(), 0.13), xytext=(-10, 5), textcoords='offset points', color='orange', fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # with zero opacity
        # plt.legend(fontsize=my_font_size)
        plt.legend(ncol=1, fontsize=my_font_size -2,
                   markerscale=3.5, framealpha=1)

        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_best_performers_{results['key']}.png") # , bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()

    @staticmethod
    def convert_to_df(results):
        # Create empty lists to store the data
        models = []
        sizes = []
        configs = []
        accuracies = []
        precisions = []
        weighted_scores = []
        model_classes = []

        # Iterate over the dictionary and extract the data
        for model, model_data in results.items():
            if model == "key":
                continue

            # don't show baseline models
            if model.split("-")[0] in ["majority_full_name", "random_full_name"]:
                continue

            accuracies.append(model_data['paraphrased']["accuracy"])
            sizes.append(model_data['size'])
            models.append(model)
            configs.append('paraphrased')
            precisions.append(model_data['paraphrased']["precision"])
            model_classes.append(model.split("-")[0])
            if "weighted_score" in model_data['paraphrased']:
                weighted_scores.append(model_data['paraphrased']["weighted_score"])
            else:
                logging.warning(f"no weighted score for {model}")
                weighted_scores.append(0)
                

        # # Create a DataFrame from the extracted data
        return pd.DataFrame({"model": models, "model_class": model_classes, "size": sizes, "precision": precisions,
                             "accuracy": accuracies, "weighted_score": weighted_scores, "config": configs})


    def plot_accuracy_overview(self, results):
        df2 = PrecomputedPlotting.convert_to_df(results)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df2, x="size", y="weighted_score", hue="model", s=300, markers=True, legend=False)

        # Prepare the labels and positions
        labels = df2['model'].tolist()
        positions = df2[['size', 'weighted_score']].values.tolist()

        # Add labels with adjustment to avoid overlap
        texts = []
        for label, position in zip(labels, positions):
            label_length = len(label)
            mv_left = label_length * 2
            # text = plt.annotate(label, position, xytext=(-mv_left, 15), textcoords='offset points', fontsize='large')
            # texts.append(text)

        # Adjust the positions of labels to prevent overlap
        # adjust_text(texts)

        # baselines hardcoded (TODO: make dynamic or change if test set changes)
        plt.axhline(y=self.baseline_random, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=self.baseline_majority, color='orange', linewidth=2.5, label="majority names")

        plt.legend(fontsize="xx-large")

         # Set x-axis to logarithmic scale
        plt.xscale('log')

        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize="xx-large")
        plt.ylabel("W-PNMS", fontsize="xx-large")

        # Increase font size of tick labels
        plt.xticks(fontsize='xx-large')
        plt.yticks(fontsize='xx-large')

        plt.title("accuracy compared to model size for paraphrased texts", fontsize="xx-large")
        plt.grid(True)
        plt.savefig(f"evaluation/plotting/plots/plot_accuracies_{results['key']}.png", bbox_inches='tight')
        # ensure pyplot does not run out of memory when too many plots are created
        plt.close()

    def plot_model_types_comparison_scatter(self, results, df):
        # add slighly larger fontsize
        my_font_size = 12
        # Dict
        model_types = {
            "bert": "fill_mask",
            "bloom": "text_generation",
            "bloomz": "text_generation",
            "cerebras": "text_generation",
            "deberta": "fill_mask",
            "deberta_squad": "question_answering",
            "distilbert_squad": "question_answering",
            "distilbert": "fill_mask",
            "falcon": "text_generation",
            "falcon_instruct": "text_generation",
            "flan_t5": "text_generation",
            "gptj": "text_generation",
            "incite_instruct": "text_generation",
            "llama": "text_generation",
            "mpt": "text_generation",
            "mt0": "text_generation",
            "mt5": "text_generation",
            "pythia": "text_generation",
            "roberta_squad": "question_answering",
            "t5": "text_generation",
            "roberta": "fill_mask",
            "gpt3.5turbo": "text_generation",
            "gpt_4": "text_generation",
        }

        # Convert dict to DataFrame
        model_types_df = pd.DataFrame(list(model_types.items()), columns=['model_class', 'model_type'])

        # Merge dataframes on model_class
        df = pd.merge(df, model_types_df, on='model_class')

        # Plot
        plt.figure(figsize=(10, 8))
        print(df)

        # group it
        df = df.sort_values('size', ascending=False)
        grouped_data = df.groupby('model_type').apply(lambda x: x.sort_values('size')).reset_index(drop=True)

        # Get unique model groups and their corresponding colors
        unique_groups = grouped_data['model_type'].unique()
        color_palette = sns.color_palette('tab10', n_colors=len(unique_groups))

        my_font_size = 24

        # scatter points
        scatter = sns.scatterplot(data=grouped_data, x='size', y='weighted_score', hue='model_type', s=250, markers=True, legend=True)
        handles, labels = scatter.get_legend_handles_labels()

        # baselines
        plt.axhline(y=self.baseline_random, color='blue', linewidth=2.5, label="random names", zorder=0)
        plt.axhline(y=self.baseline_majority, color='orange', linewidth=2.5, label="majority names", zorder=0)

        # Create custom legend entries
        import matplotlib.lines as mlines
        random_line = mlines.Line2D([], [], color='blue', marker='_', markersize=15, label='random names baseline', linewidth=1.5)
        majority_line = mlines.Line2D([], [], color='orange', marker='_', markersize=15, label='majority names baseline', linewidth=1.5)


        # handles.extend([random_patch, majority_patch])
        handles.extend([random_line, majority_line])
        labels.extend(['random names', 'majority names'])

        plt.legend(handles=handles, labels=labels)

        # Plot separate lines for each group with corresponding colors
        for group, color in zip(unique_groups, color_palette):
            group_data = grouped_data[grouped_data['model_type'] == group]
            plt.plot(group_data['size'], group_data['weighted_score'], color=color, label="_nolegend_", linewidth=3)

        plt.legend(fontsize="x-large")
        plt.xlabel('model size [million parameters]', fontsize="x-large")
        plt.ylabel('weighted partial name match score', fontsize="x-large")
        plt.title('Model Performance by Type', fontsize="x-large")
        plt.savefig(f"evaluation/plotting/plots/ablations/toto_plot_model_types_comparison_scatter_{results['key']}.png")

    
    def plot_model_types_comparison(self, results, df):
        model_types = {
            "bert": "Fill Mask",
            "bloom": "Text Generation",
            "bloomz": "Text Generation",
            "cerebras": "Text Generation",
            "deberta": "Fill Mask",
            "deberta_squad": "Question Answering",
            "distilbert_squad": "Question Answering",
            "distilbert": "Fill Mask",
            "falcon": "Text Generation",
            "falcon_instruct": "Text Generation",
            "flan_t5": "Text Generation",
            "gptj": "Text Generation",
            "incite_instruct": "Text Generation",
            "llama": "Text Generation",
            "mpt": "Text Generation",
            "mt0": "Text Generation",
            "mt5": "Text Generation",
            "pythia": "Text Generation",
            "roberta_squad": "Question Answering",
            "t5": "Text Generation",
            "roberta": "Fill Mask",
            "gpt3.5turbo": "Text Generation",
            "gpt_4": "Text Generation",
        }

        # Convert dict to DataFrame
        model_types_df = pd.DataFrame(list(model_types.items()), columns=['model_class', 'model_type'])

        # exlucde very bad text generation models, they do not represent the class well
        excluded_models = [
            "pythia", "cerebras", "falcon_instruct", "gptj", "falcon"
        ]
        # filter out rows which have model_class in exluded_models
        df = df[~df["model_class"].isin(excluded_models)]

        print(df)

        # Merge dataframes on model_class
        df = pd.merge(df, model_types_df, on='model_class')

        # neclect all models that performed below the baseline and are text generation models
        # df = df[~((df['weighted_score'] < self.baseline_random) & (df['model_type'] == "Text Generation"))]

        # upscale billions to normal numbers
        df['size'] = df['size'] * 1e9

        plt.figure(figsize=(9, 4))
        # y axis weighted_score, x axis is model size, color is model type
        scatter = sns.scatterplot(data=df, x="size", y="weighted_score", hue="model_type", s=250, markers=True, legend=True)

        # Set x-axis to logarithmic scale
        plt.xscale('log')

        plt.ylabel("W-PNMS", fontsize="xx-large")
        plt.xlabel('Size [Parameters]', fontsize="xx-large")
        plt.tick_params(axis='both', which='major', labelsize="xx-large")
        # Enlarging the legend dots
        legend_labels, _= scatter.get_legend_handles_labels()
        scatter.legend(legend_labels, df["model_type"].unique(), title='', loc='upper left',
                       markerscale=1.5, fontsize="x-large", framealpha=1)
        plt.grid()
        plt.savefig(f"evaluation/plotting/plots/ablations/plot_model_types_comparison_{results['key']}.png",
                    dpi=300, bbox_inches='tight')


    def plot_accuracy_overview_with_legend_and_size(self, results, df2):
        print(df2.head())
        plt.figure(figsize=(20, 14))

        my_font_size = 24
        sns.scatterplot(data=df2, x="size", y="weighted_score", hue="model_class", s=350, markers=True)

        # baselines
        plt.axhline(y=self.baseline_random, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=self.baseline_majority, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize=my_font_size)
        plt.ylabel("Accuracy", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        plt.legend(ncol=1, fontsize=my_font_size -2,
                   markerscale=3.5, framealpha=1, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True)

        plt.savefig(f"evaluation/plotting/plots/plot_accuracies_with_legend_and_size_{results['key']}.png") #, bbox_inches='tight')


    @staticmethod
    def plot_normal_to_instructional(results, df2):
        """expect df2 to contain each model twice, once for every compared input size"""
        plt.figure(figsize=(20, 14))

        # expect the following models
        models = ["falcon", "roberta", "distilbert", "mpt", "incite", "t5"]

        df2["is_instructional"] = df2["model_class"].apply(
            lambda x: "instruction tuned" if ("squad" in x or "flan" in x or "instruct" in x) else "base")

        my_font_size = 24
        sns.scatterplot(data=df2, x="size", y="weighted_score", hue="is_instructional", s=350, markers=True)

        for i, row in df2.iterrows():
            label_length = len(row['model'])
            # if the point is to the right of the plot, move the label to the left
            if row['size'] == df2['size'].max():
                mv_left = - (label_length * 10.5)
            # small hack for gptj, to make it fit as well
            else:
                mv_left = 12
            plt.annotate(row['model_class'], (row['size'], row['weighted_score']), xytext=(mv_left, -7), textcoords='offset points',
                         fontsize=my_font_size)

        grouped_data = df2.groupby(df2["model"].str.extract(f"({'|'.join(models)})")[0])

        for group, group_data in grouped_data:

            # Sort the group data by 'size' in ascending order
            sorted_data = group_data.sort_values('is_instructional')
            
            # Extract the x and y values for the two points
            x1, y1 = sorted_data.iloc[0]['size'], sorted_data.iloc[0]['weighted_score']
            x2, y2 = sorted_data.iloc[1]['size'], sorted_data.iloc[1]['weighted_score']
            
            # padding to not overlap labels
            padding = 0.004
            # Calculate the difference in y values
            y_diff = abs(y2 - y1) - padding - 0.004 # remove dot radius

            start = min(y1, y2) + padding
            
            # Draw the arrow from normal to instructional
            plt.arrow(x1, start, 0, y_diff, head_width=0.1, head_length=0.003, width=0.015, color='black', length_includes_head=True)
            
            # Add the percentage score alongside the arrow, move text by "percentage_length" to the right
            percentage_length = 0.75
            plt.text(x1 + percentage_length, start + y_diff / 2, f'+{abs(y_diff):.2%}', ha='right', va='center', fontsize=my_font_size - 2)

        
        # Set labels and title
        plt.xlabel("Size [Billion Parameters]", fontsize=my_font_size)
        plt.ylabel("Accuracy", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        max_x = df2['size'].max()
        plt.xlim(None, max_x + percentage_length)


        legend = plt.legend(ncol=1, fontsize=my_font_size -2,
                   markerscale=3.5, framealpha=1)

        plt.savefig(f"evaluation/plotting/plots/ablations/plot_normal_to_instructional_{results['key']}.png", bbox_inches='tight')

    @staticmethod
    def plot_normal_to_instructional_barplot(results, df):
        """expect df to contain each model twice, once for every compared input size"""
        plt.figure(figsize=(20, 8))
        my_font_size = 42

        # expect the following models
        models = ["falcon", "roberta", "distilbert", "mpt", "incite", "t5", "bloom"]

        names = {
            "bloomz": "BLOOMZ",
            "flan_t5": "Flan_T5",
            "roberta": "RoBERTa",
            "t5": "T5",
            "mt0": "mT0",
            "bloom": "BLOOM",
            "cerebras": "Cerebras-GPT",
            "pythia": "Pythia",
            "t0": "T0",
            "falcon": "Falcon",
            "mpt": "MPT",
            "distilbert": "DistilBERT",
            "incite_instruct": "INCITE-Instruct",
            "incite": "INCITE",
            "roberta_squad": "RoBERTa-SQuAD",
        }


        # Group by model names
        df['model_name'] = df["model"].str.extract(f"({'|'.join(models)})")[0]
        # replace model_class names with name from the names dict
        df['model_name'] = df['model_name'].replace(names)

        df["is_instructional"] = df["model_class"].apply(
            lambda x: "instruction tuned" if ("squad" in x or "flan" in x or "instruct" in x) else "base")
        

        grouped = df.groupby(['model_name', 'is_instructional'])
        size_grouped = df.groupby('model_name')['size'].mean()

        # Prepare data for plot
        plot_data = []
        for name, group in grouped:
            plot_data.append((name[0], name[1], group['weighted_score'].mean()))

        # Sort data by score
        plot_data.sort(key=lambda x: x[2])

        # Split data into separate lists
        names, types, accuracies = zip(*plot_data)
        unique_names = sorted(set(names))
        model_sizes = df.groupby('model_name')['size'].unique()

        # Create x-axis tick labels with size information
        unique_names_with_size = []
        sizes_in_billion = []
        for name in unique_names:
            sizes = []
            for size in model_sizes[name]:
                if size < 1:
                    # Format as millions with non-zero decimals
                    sizes.append(f'{size*1000:.0f}M' if size*1000 % 1 == 0 else f'{size*1000:.1f}M')
                else:
                    # Format as billions with non-zero decimals
                    sizes.append(f'{size:.0f}B' if size % 1 == 0 else f'{size:.1f}B')
            unique_names_with_size.append(f'{name}\n{"/".join(sizes)}')
            sizes_in_billion.append(float(sizes[0][:-1]) if sizes[0][-1] == 'B' else float(sizes[0][:-1]) / 1000)

        # Combine the names and sizes, and sort them by size
        names_and_sizes = sorted(zip(unique_names_with_size, sizes_in_billion), key=lambda x: x[1])

        # Get the sorted names
        unique_names_with_size = [name for name, size in names_and_sizes]

        normal_accuracies = [accuracy for name, type_, accuracy in plot_data if type_ == 'base']
        instruction_accuracies = [accuracy for name, type_, accuracy in plot_data if type_ == 'instruction tuned']

        # Calculate the width of a bar
        bar_width = 0.3

        # Positions of the left bar boundaries
        bar_l = np.arange(len(unique_names))

        # Positions of the x-axis ticks (center of the bars as bar labels)
        tick_pos = [i + bar_width / 2 for i in bar_l]

        # Create the bar plot
        plt.bar(bar_l, normal_accuracies, width=bar_width, label='base', color='orange', zorder=1)
        plt.bar(bar_l + bar_width, instruction_accuracies, width=bar_width, label='instructional', color='blue', zorder=1)

        # Set the labels and title
        plt.ylabel('W-PNMS', fontsize=my_font_size)

        # Set the positions and labels of the x-axis ticks
        plt.xticks(tick_pos, unique_names_with_size)
        
        # increase height of plot so legend fits
        y_max = max(max(normal_accuracies), max(instruction_accuracies))
        plt.ylim(0, y_max + 0.07)

        # Adding the legend and showing the plot
        plt.legend(fontsize=my_font_size)
        
        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        plt.grid(axis="y", zorder=0)

        plt.savefig(f"evaluation/plotting/plots/ablations/plot_normal_to_instructional_barplot_{results['key']}.png", bbox_inches='tight')

    @staticmethod
    def plot_accuracy_input_size_comparison(results, df2):
        """expect df2 to contain each model twice, once for every compared input size"""
        plt.figure(figsize=(20, 14))

        # use input size as hue
        df2['input_size'] = df2['model_class'].apply(lambda x: x.split("_")[-1])
        # remove input sizes from model class names
        df2['model_class'] = df2['model_class'].apply(lambda x: "_".join(x.split("_")[:-1]))

        my_font_size = 18
        df3 = df2.sort_values('input_size', ascending=False)
        sns.scatterplot(data=df3, x="size", y="accuracy", hue="input_size", s=350, markers=True)

        grouped_data = df2.groupby('model_class')

        for group, group_data in grouped_data:

            # Sort the group data by 'size' in ascending order
            sorted_data = group_data.sort_values('size')
            
            # Extract the x and y values for the two points
            x1, y1 = sorted_data.iloc[0]['size'], sorted_data.iloc[0]['accuracy']
            x2, y2 = sorted_data.iloc[1]['size'], sorted_data.iloc[1]['accuracy']
            
            # padding to not overlap labels
            padding = 0.008
            # Calculate the difference in y values
            y_diff = abs(y2 - y1) - 0.006 # remove dot radius

            start = min(y1, y2) + 0.003
            if len(sorted_data) == 2:
                start += padding
                y_diff -= padding
            
            # Draw the arrow
            plt.arrow(x1, start, 0, y_diff, head_width=0.1, head_length=0.003, width=0.015, color='black', length_includes_head=True)
            
            # Add the percentage score alongside the arrow, move text by "percentage_length" to the right
            percentage_length = 0.86
            plt.text(x1 + percentage_length, start + y_diff / 2, f'+{abs(y_diff):.2%}', ha='right', va='center', fontsize=my_font_size - 2)

            # got a third point, compare it as well
            if len(sorted_data) == 3:
                x3, y3 = sorted_data.iloc[2]['size'], sorted_data.iloc[2]['accuracy']

                # Calculate the difference in y values
                y_diff = abs(y3 - y1) - 0.006 # remove dot radius

                start = min(y1, y3) + 0.003

                # Draw the arrow
                plt.arrow(x2, start, 0, y_diff, head_width=0.1, head_length=0.003, width=0.015, color='black', length_includes_head=True)

                # Add the percentage score alongside the arrow, move text by "percentage_length" to the right
                percentage_length = 0.8
                plt.text(x2 + percentage_length, start + y_diff / 2, f'+{abs(y_diff):.2%}', ha='right', va='center', fontsize=my_font_size - 2)

                # add the model label
                entry = sorted_data.iloc[2]
                label_length = len(entry['model'])
                # only label the top-most
                mv_left = label_length * 2 + -5
                plt.annotate(entry['model_class'], (entry['size'], entry['accuracy']), xytext=(-mv_left, 12), textcoords='offset points',
                            fontsize=my_font_size)
            else:
                # add the model label
                entry = sorted_data.iloc[1]
                label_length = len(entry['model'])
                # only label the top-most
                mv_left = label_length * 2 + -5
                plt.annotate(entry['model_class'], (entry['size'], entry['accuracy']), xytext=(-mv_left, 12), textcoords='offset points',
                            fontsize=my_font_size)

        # Set labels and title
        plt.xlabel("size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("partial name match score", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        max_x = df2['size'].max()
        plt.xlim(None, max_x + 1)

        legend = plt.legend(ncol=1, fontsize=my_font_size -2,
                   markerscale=3.5, framealpha=1)
        legend.set_title("input size [chars]", prop={'size': my_font_size})

        plt.savefig(f"evaluation/plotting/plots/plot_input_length_comparison_{results['key']}.png")
    
    @staticmethod
    def plot_accuracy_overview_with_legend(results, df2):
        plt.figure(figsize=(20, 14))

        my_font_size = 24

        sns.scatterplot(data=df2, x="size", y="accuracy", hue="model", s=350, markers=True)

        # baselines
        plt.axhline(y=0.06, color='blue', linewidth=2.5, label="random names")
        plt.axhline(y=0.13, color='orange', linewidth=2.5, label="majority names")

        # Set labels and title
        plt.xlabel("size [billion parameters]", fontsize=my_font_size)
        plt.ylabel("partial name match score", fontsize=my_font_size)

        # Increase font size of tick labels
        plt.xticks(fontsize=my_font_size)
        plt.yticks(fontsize=my_font_size)

        # Set x-axis to logarithmic scale
        plt.xscale('log')

        # plt.title("accuracy compared to model size for paraphrased texts", fontsize=my_font_size + 10)

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=2,fontsize=my_font_size,
                   markerscale=3.5)
        plt.grid(True)

        plt.savefig(f"evaluation/plotting/plots/plot_accuracies_with_legend_{results['key']}.png", bbox_inches='tight')

    @staticmethod
    def tabulate_results_to_latex(results):
        df2 = PrecomputedPlotting.convert_to_df(results)        
        # Keep only the specified columns
        df2 = df2[['model_class', 'size', 'accuracy', 'precision', 'weighted_score']]
        
        df2 = df2.sort_values(by=['weighted_score'], ascending=False)
        # round and reduce accuracy to 2 digits
        df2['accuracy'] = df2['accuracy'].apply(lambda x: round(x, 2))
        df2['size'] = df2['size'].apply(lambda x: round(x, 2))
        df2['precision'] = df2['precision'].apply(lambda x: round(x, 2))
        df2['weighted_score'] = df2['weighted_score'].apply(lambda x: round(x, 2))

        # Rename the columns
        df2 = df2.rename(columns={'size': 'size [billions]', 'accuracy': 'partial name matching score',
                                  'precision': 'levenshtein distance','weighted_score': 'weighted score'})
    
        # Ensure the index is of string type
        df2.index = df2.index.astype(str)
        
        # Now you can replace underscores
        df2.index = df2.index.str.replace('_', '\\_')
        
        latext_text = df2.to_latex(index=False, float_format="{:0.2f}".format)
        with open(f"evaluation/plotting/plots/latex_table_{results['key']}.txt", "w") as f:
            f.write(latext_text)


def main():
    if len(sys.argv[1]) > 0:
        print(sys.argv[1])
        PrecomputedPlotting(sys.argv[1]).plot()
    else:
        PrecomputedPlotting("run-all-top-5").plot()


if __name__ == '__main__':
    main()
