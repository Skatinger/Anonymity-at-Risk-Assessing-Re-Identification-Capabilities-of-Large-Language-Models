# small helper to create shards for the parallelization of the paraphrasing
from datasets import load_from_disk

dataset = load_from_disk("./data_unparaphrased")

numShards = 8
for i in range(numShards):
    shard = dataset.shard(num_shards=numShards, index=i)
    shard.save_to_disk("./data_unparaphrased_shard_{}".format(i))
