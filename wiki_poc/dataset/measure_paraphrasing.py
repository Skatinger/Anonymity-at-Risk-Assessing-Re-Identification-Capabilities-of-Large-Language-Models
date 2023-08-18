""" measures how much the dataset was changed from original to paraphrased version"""


from datasets import load_dataset
# from ..models.model_runner import load_test_set
import Levenshtein

dataset = load_dataset("Skatinger/wikipedia-persons-masked", split="train")

def compare_distances(example):
    # compare sentence by sentence, compute average distance over all sentences
    distances = []
    original_sentence_length = []
    paraphrased_sentence_length = []
    for original_sentence, paraphrased_sentence in zip(example["sentences"], example["paraphrased_sentences"]):
        distance = Levenshtein.distance(original_sentence, paraphrased_sentence)
        distances.append(distance)
        original_sentence_length.append(len(original_sentence))
        paraphrased_sentence_length.append(len(paraphrased_sentence))
    example["mean_distance"] = sum(distances) / len(distances)
    example["distances"] = distances
    example["original_sentence_length"] = sum(original_sentence_length) / len(original_sentence_length)
    example["paraphrased_sentence_length"] = sum(paraphrased_sentence_length) / len(paraphrased_sentence_length)
    return example

dataset = dataset.map(compare_distances, num_proc=8)

mean = sum(dataset["mean_distance"]) / len(dataset["mean_distance"])
print(f"mean distance: {mean}")
print(f"min distance: {min(dataset['mean_distance'])}")
print(f"max distance: {max(dataset['mean_distance'])}")
# average sentences length
print(f"average original sentence length: {sum(dataset['original_sentence_length']) / len(dataset['original_sentence_length'])}")
print(f"average paraphrased sentence length: {sum(dataset['paraphrased_sentence_length']) / len(dataset['paraphrased_sentence_length'])}")
