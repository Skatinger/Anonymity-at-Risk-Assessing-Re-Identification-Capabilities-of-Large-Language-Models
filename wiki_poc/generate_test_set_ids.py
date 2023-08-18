
from datasets import load_dataset
import os
import sys

"""
creates a file models/test_set_ids{_rulings}.csv with the ids of the test set of the wikipedia dataset
ids are sharded from the base dataset, identifying 10'000 pages.
"""
def main():
    assert len(sys.argv) == 2, "Please specify dataset as argument. Options: wiki, rulings"
    dataset = sys.argv[1]
    if dataset == "wiki":
        print("Preparing test set ids for wikipedia...")
        # ensure file dos not exist, as we dont want to overwrite it
        assert os.path.exists("test_set_ids.csv") == False, "File already exists, remove it to generate a new one."
        base = load_dataset("Skatinger/wikipedia-persons-masked", split="train")
        factor = len(base) / 10000
        shard = base.shard(factor, 0)

        with open("models/test_set_ids.csv", "w") as f:
            for i in shard:
                f.write(i["id"] + "\n")
    
    elif dataset == "rulings":
        print("Preparing test set ids for rulings...")
        # ensure file dos not exist, as we dont want to overwrite it
        assert os.path.exists("test_set_ids_rulings.csv") == False, "File already exists, remove it to generate a new one."
        rulings = load_dataset("rcds/swiss_rulings", split="train")
        bger_rulings = rulings.filter(lambda x: x["court"] == "CH_BGer")
        recent_rulings = bger_rulings.filter(lambda x: x["year"] > 2018)
        # reduce to 10'000 rulings
        factor = len(recent_rulings) / 10000
        shard = recent_rulings.shard(factor, 0)

        with open("test_set_ids_rulings.csv", "w") as f:
            for i in shard:
                f.write(i["decision_id"] + "\n")
    
    else:
        print("Dataset not found. Please specify either wiki or rulings as argument.")


if __name__ == "__main__":
    main()