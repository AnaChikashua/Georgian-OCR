import datasets
from datasets.tasks import ImageClassification
# _NAMES = ['ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ', 'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ', 'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ', 'ხ', 'ჯ', 'ჰ']
logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {Georgian language alphabet dataset},
author={Ana Chikashua},
year={2023}
}
"""
_DESCRIPTION = """
Georgian language handwriting dataset!
"""
_URL = "https://huggingface.co/datasets/AnaChikashua/handwriting/resolve/main/handwriting_dataset.zip"


class HandwritingData(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                 "label": datasets.features.ClassLabel(),
                 "image": datasets.Image()
                 }
            ),
            supervised_keys=("image", "label"),
            homepage="https://huggingface.co/datasets/AnaChikashua/alphabet",
            # task_templates=[ImageClassification(image_column="image", label_column="label")],

        )

    def _split_generators(self, dl_manager):
        path = dl_manager.dowload(_URL)
        image_iters = dl_manager.iter_archive(path)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"images": image_iters}
            ),
        ]

    def _generate_examples(self, images):
        """This function returns the examples in the raw (text) form."""
        # Iterate through images
        for idx, filepath, image in enumerate(images):
            # extract the text from the filename
            logger.error(filepath)
            text = [c for c in str(filepath) if not 0 <= ord(c) <= 127][0]
            yield idx, {
                "label": str(idx)+'txt',
                "image": image
            }
