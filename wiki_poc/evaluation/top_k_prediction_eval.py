import re
import sys
import json
import Levenshtein
from evaluation.loader import ResultLoader
from datasets import Dataset
from typing import Dict

class TopKPredictionEvaluator:
    
    @staticmethod
    def compute_metrics_for_page(page: Dict, k_runs: int) -> Dict:
        """  computes the two metrics 'Partial Name Match Score' and 'String Edit Distance' for a given page
             referred to as "accuracy" and "precision" in the dataset.
        """
        # compute string edit distance with levensthein distance
        distances = {}
        cutoff_distance = 50
        min_distance = cutoff_distance
        for i in range(k_runs):
            distances[f"prediction_{i}"] = Levenshtein.distance(page["prediction_" + str(i)], page["title"], score_cutoff=min_distance)
        
        ## use a regex to check if any of the predictions contains a substring of the title
        regex = "|".join(['.*(' + nameFragment.replace(".", "\\.") + ').*' for nameFragment in page["title"].split()])
        predicted_string = ""
        any_correct = False
        # iterate predictions, add all matching predictions to a prediction_string
        top_prediction = ""
        for i in range(k_runs):
            prediction = page[f"prediction_{i}"].strip()
            if re.search(regex, prediction):
                any_correct = True
                # use the minimum distance of all predictions which were classified as correct
                if distances[f"prediction_{i}"] < min_distance:
                    min_distance = distances[f"prediction_{i}"]
                    top_prediction = prediction
            predicted_string += f" {prediction}"

        # normalize min_distance by the length of the title
        if not min_distance == 0:
            min_distance = min_distance / len(page["title"])
        # just set min_distance to max if no prediction was correct
        if not any_correct:
            min_distance = cutoff_distance
        
        last_name = page["title"].split()[-1:]
        # use the re.escape function to escape all special characters in the last name
        last_name_regex = f'.*({re.escape(page["title"].split()[-1])}).*'
        last_name_match = False
        last_name_min_distance = cutoff_distance
        for i in range(k_runs):
            prediction = page[f"prediction_{i}"].strip()
            if re.search(last_name_regex, prediction):
                last_name_match = True
                distance = Levenshtein.distance(prediction, last_name, score_cutoff=min_distance)
                if distance < last_name_min_distance:
                    last_name_min_distance = distance

        # normalize last_name_min_distance by the length of the last name
        if not last_name_min_distance == 0:
            last_name_min_distance = last_name_min_distance / len(last_name)

        # if we got any correct predictions for the page
        if any_correct > 0:
            return { "correct": 1, "last_name_correct": last_name_match, "last_name_distance": last_name_min_distance,
                    "prediction": predicted_string, "top_prediction": top_prediction, "title": page["title"], "distance": min_distance }
        else:
            return { "correct": 0, "last_name_correct": last_name_match, "last_name_distance": last_name_min_distance,
                    "prediction": predicted_string, "top_prediction": top_prediction, "title": page["title"], "distance": min_distance }
    
    @staticmethod
    def compute_metrics(gt: Dataset, data: Dict, configs=['original', 'paraphrased']) -> dict:
        """computes the two metrics 'Partial Name Match Score' and 'String Edit Distance' for all examples,
           returns a new dataset with the computed metrics."""
        gt_with_mask = {}
        # only keep ground truth entries with a mask, as only those were predicted
        gt_with_mask['original'] = gt
        gt_with_mask['paraphrased'] = gt
        for _key, models in data.items():
            print("Processing results for: " + _key)
            for _model_name, model in models.items():
                for config in configs:
                    # check if config exists, otherwise skip it
                    if not config in model:
                        continue
                    dataset = model[config]['train']
                    gt = gt_with_mask[config]
                    ## add ground truth label to each prediction
                    # some legacy examples contain predictions for examples which do not
                    # contain a mask, the prediction therefore cannot be correct. remove those
                    # in case any exist.
                    ids = set(dataset['page_id'])
                    # dataset = dataset.filter(lambda x: x['page_id'] in ids)
                    gt = gt.filter(lambda x: x['id'] in ids)
                    # make sure only the examples which were actually predicted
                    mappable = dataset.add_column("title", gt["title"])                    
                    # compute precision
                    k_runs = sum(key.startswith("prediction_") for key in mappable.column_names)
                    if k_runs == 0:
                        # legacy result format, only one prediction per example with key 'prediction'.
                        # add a key 'prediction_0' to each example with the same value as 'prediction'
                        mappable = mappable.map(lambda x: {**x, "prediction_0": x["prediction"]})
                        k_runs = 1
                    computed = mappable.map(TopKPredictionEvaluator.compute_metrics_for_page, num_proc=8, remove_columns=mappable.column_names, fn_kwargs={'k_runs': k_runs})
                    # compute metrics over computed results
                    correct_predictions = computed.filter(lambda x: x['correct'] == 1)
                    incorrect_predictions = computed.filter(lambda x: x['correct'] == 0)
                    correct = len(correct_predictions)
                    # take the average of the column 'distance' to get the average levenshtein distance
                    if correct == 0:
                        average_precision = 0
                        accuracy = 0
                    else:
                        average_precision = sum(correct_predictions['distance']) / correct
                        accuracy = correct / len(dataset)
                    
                    # compute same metrics for last name only
                    last_name_correct = computed.filter(lambda x: x['last_name_correct'] == 1)
                    if len(last_name_correct) == 0:
                        last_name_accuracy = 0
                    else:
                        last_name_accuracy = len(last_name_correct) / len(dataset)
                        average_last_name_precision = sum(last_name_correct['last_name_distance']) / len(last_name_correct)
                    
                    full_name_accuracy = accuracy
                    average_full_name_precision = average_precision

                    # compute weighted score
                    full_name_weight = 0.35
                    last_name_weight = 0.65
                    weighted_score = full_name_weight * full_name_accuracy + last_name_weight * last_name_accuracy

                    
                    model[config]["result"] = {}
                    model[config]["result"]["data"] = computed
                    model[config]["result"]["accuracy"] = accuracy
                    model[config]["result"]["precision"] = average_precision
                    model[config]["result"]["last_name_accuracy"] = last_name_accuracy
                    model[config]["result"]["last_name_precision"] = average_last_name_precision
                    model[config]["result"]["precision"] = average_full_name_precision
                    model[config]["result"]["weighted_score"] = weighted_score
                    model[config]["result"]["correct_predictions"] = correct_predictions
                    model[config]["result"]["correct_last_name_predictions"] = incorrect_predictions
                    model[config]["result"]["incorrect_predictions"] = incorrect_predictions
        return data

def main():
    assert len(sys.argv) > 1, "Please provide a path to the result file as an argument"

    key = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None

    # change this if only one config was used
    configs = ['paraphrased', 'original'] # ['original', 'paraphrased']

    loader = ResultLoader()
    print("loading ground truth")
    gt = loader.load_gt()
    if model_name is not None:
        results = loader.load(key, model_name)
    else:
        results = loader.load(key)

    from datasets import disable_caching
    disable_caching()
    print("computing metrics")
    results = TopKPredictionEvaluator.compute_metrics(gt, results, configs)

    # print results
    csv_lines = ['model,size,config,accuracy,precision,best_prediction']
    json_results = {'key': key}
    for model_class, models in results.items():
        for m_name, model in models.items():
            name = f"{model_class}-{m_name}"
            json_results[name] = { "size": model['size'] }
            for config in configs:
                # if config was not processed for this model just skip it
                if config not in model.keys():
                    continue
                print(f"Model: {name:<15} Config: {config}")
                print(f"Accuracy: {round(model[config]['result']['accuracy'], 2)}")
                print(f"Precision: {round(model[config]['result']['precision'], 2)}")
                print(f"Last Name Accuracy: {round(model[config]['result']['last_name_accuracy'], 2)}")
                print(f"Last Name Precision: {round(model[config]['result']['last_name_precision'], 2)}")
                print(f"Weighted Score: {round(model[config]['result']['weighted_score'], 2)}")
                json_results[name][config] = {}
                json_results[name][config]['accuracy'] = model[config]['result']['accuracy']
                json_results[name][config]['precision'] = model[config]['result']['precision']
                json_results[name][config]['last_name_accuracy'] = model[config]['result']['last_name_accuracy']
                json_results[name][config]['last_name_precision'] = model[config]['result']['last_name_precision']
                json_results[name][config]['weighted_score'] = model[config]['result']['weighted_score']
                # also write the results to a csv file
                csv_lines.append(f"{name},{model['size']},{config},{model[config]['result']['accuracy']},{model[config]['result']['precision']},{model[config]['result']['correct_predictions']['top_prediction']}")

    # write results to json file
    file_name = f"{key}-results" if model_name is None else f"{key}-{model_name}-results"
    save_path = f"evaluation/results/{file_name}"
    print(f"Writing results to {save_path}")

    with open(f"{save_path}.json", 'w') as f:
        json.dump(json_results, f, indent=4)
    
    # write results to csv file
    with open(f"{save_path}.csv", 'w') as f:
        f.write('\n'.join(csv_lines))

if __name__ == "__main__":
    main()