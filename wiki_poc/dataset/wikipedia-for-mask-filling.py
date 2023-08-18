# dataset loading script for huggingface
import datasets
import json
try:
    import lzma as xz
except ImportError:
    import pylzma as xz

datasets.logging.set_verbosity_info()
logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """\
    """

_HOMEPAGE = "https://skatinger.github.io/master_thesis/",

_LICENSE = ""

_CITATION = ""

_TYPES = ["original", "paraphrased"]

_SIZES = [4096, 512]

_URLS = {
    "original_4096": "data/original_4096.jsonl.xz",
    "original_512": "data/original_512.jsonl.xz",
    "paraphrased_4096": "data/paraphrased_4096.jsonl.xz",
    "paraphrased_512": "data/paraphrased_512.jsonl.xz"
}


class WikipediaForMaskFillingConfig(datasets.BuilderConfig):
    """BuilderConfig for WikipediaForMaskFilling.

    features: *list[string]*, list of the features that will appear in the
    feature dict. Should not include "label".
    **kwargs: keyword arguments forwarded to super
    """

    def __init__(self, type, size=4096, **kwargs):
        """BuilderConfig for WikipediaForMaskFilling.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """

        # Version history:
        # 1.0.0: first version
        super().__init__(**kwargs)
        self.size = size
        self.type = type


class WikipediaForMaskFilling(datasets.GeneratorBasedBuilder):
    """WikipediaForMaskFilling dataset."""

    BUILDER_CONFIGS = [
        WikipediaForMaskFillingConfig(
            name="original_4096",
            version=datasets.Version("1.0.0"),
            description="Part of the dataset with original texts and masks, with text chunks split into size of max 4096 tokens (Longformer).",
            max_tokens=4096,
            type="original"
        ),
        WikipediaForMaskFillingConfig(
            name="original_512",
            version=datasets.Version("1.0.0"),
            description="text chunks split into size of max 512 tokens (roberta).",
            max_tokens=512,
            type="original"
        ),
        WikipediaForMaskFillingConfig(
            name="paraphrased_4096",
            version=datasets.Version("1.0.0"),
            description="Part of the dataset with paraphrased texts and masks, with text chunks split into size of max 4096 tokens (Longformer).",
            max_tokens=4096,
            type="paraphrased"
        ),
        WikipediaForMaskFillingConfig(
            name="paraphrased_512",
            version=datasets.Version("1.0.0"),
            description="Paraphrased text chunks split into size of max 512 tokens (roberta).",
            max_tokens=512,
            type="paraphrased"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "texts": datasets.Value("string"),
                    "masks": datasets.Sequence(datasets.Value("string")),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        type = self.config.type
        size = self.config.size
        urls_to_download = f"data/{type}_{size}.jsonl.xz"
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name='train', gen_kwargs={"filepath": downloaded_files["train"]}),
        ]

    def _generate_examples(self, filepath):
        _id = 0
        with open(filepath, encoding="utf-8") as f:
            try:
                with xz.open(filepath) as f:
                    for line in f:
                        data = json.loads(line)
                        yield _id, {
                            "texts": data["texts"],
                            "masks": data["masks"]
                        }
                        _id += 1
            except Exception:
                logger.exception("Error while processing file %s", filepath)
