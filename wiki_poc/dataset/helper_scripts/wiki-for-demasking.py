from datasets import load_dataset
import sys
sys.path.append("..")
from custom.splitter import Splitter
from transformers import LongformerTokenizer
from transformers import AutoTokenizer


# helper function to split the examples into chunks around the masks
# and save as a new dataset, with just sentences with a single mask token and the belonging
# prediction.
# already processed version available on huggingface on rcds/wikipedia-for-fill-mask
# this script only creates a jsonl file. Before uploading to huggingface, compress it to xz
# with `xz -T4 -9e for_mask_prediction.jsonl`.

dataset = load_dataset("rcds/wikipedia-persons-masked", split='train')

### now testing this batch version
def split_examples(examples, type, size, tokenizer):
    # lists which will contain the examples after splitting, returned as a dict of lists
    all_text_chunks = []
    all_mask_chunks = []
    all_ids = []
    all_titles = []
    all_sequence_numbers = []

    # prepare the data for splitting
    content = zip(
        examples['id'],
        examples['title'],
        examples["masked_text_{}".format(type)],
        examples["masked_entities_{}".format(type)]
    )
    for id, title, text, masks in content:
        # create chunks of text for every example, add list of belonging attributes to the lists which will be returned
        # as a dict of lists (e.g. the list of new examples)
        text_chunks = [*Splitter.split_by_max_tokens(text, tokenizer, max_tokens=tokenizer.model_max_length)]
        # for each text_chunk, get the number of masks in it, and get the corresponding masks
        last_chunk_mask_index = 0
        for chunk in text_chunks:
            # get the number of masks in the chunk
            num_masks = chunk.count('<mask>')
            # get the corresponding masks
            chunk_masks = masks[last_chunk_mask_index:(num_masks + last_chunk_mask_index)]
            # update the last_chunk_mask_index
            last_chunk_mask_index += num_masks
            # add the masks to the mask_chunks
            all_mask_chunks += [chunk_masks]

        # add the newly created chunks to the list of all chunks
        all_text_chunks += text_chunks
        # add metadata to the lists
        nb_chunks_for_example = len(text_chunks)
        all_ids += [id] * nb_chunks_for_example
        all_titles += [title] * nb_chunks_for_example
        all_sequence_numbers += list(range(nb_chunks_for_example))

    # define lists for metadata which stays the same for all examples in the batch
    all_types = [type] * len(all_text_chunks)
    all_sizes = [size] * len(all_text_chunks)

    return {
        "id": all_ids,  # id of the original wiki page
        "sequence_number": all_sequence_numbers,  # sequence number of the chunk in the original wiki page
        "title": all_titles,  # title of the original wiki page e.g. the entity
        "type": all_types,  # type of the base text, either original or paraphrased
        "size": all_sizes,  # size of the base text in tokens, either 4096 or 512
        "texts": all_text_chunks,  # the text chunks
        "masks": all_mask_chunks  # the masks for the text chunks
        }


longformer_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
xlm_roberta_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

configs = [
    {'type': 'original', 'size': 4096, 'tokenizer': longformer_tokenizer},
    {'type': 'paraphrased', 'size': 4096, 'tokenizer': longformer_tokenizer},
    {'type': 'original', 'size': 512, 'tokenizer': xlm_roberta_tokenizer},
    {'type': 'paraphrased', 'size': 512, 'tokenizer': xlm_roberta_tokenizer}
]

for config in configs:
    configured_dataset = dataset.map(split_examples, batched=True, batch_size=128, remove_columns=dataset.column_names, num_proc=6, fn_kwargs=config)
    configured_dataset = configured_dataset.filter(lambda row: '<mask>' in row['texts'])
    configured_dataset.to_json('{}_{}.jsonl'.format(config['type'], config['size']))
