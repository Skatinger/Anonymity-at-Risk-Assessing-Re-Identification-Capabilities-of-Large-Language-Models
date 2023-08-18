# small helper to reunite the shards after the paraphrasing
from datasets import load_from_disk, concatenate_datasets

shards = [load_from_disk("./data_paraphrased_shard_{}/".format(i)) for i in range(8)]

dataset = concatenate_datasets(shards)

dataset.save_to_disk("./data_paraphrased")
