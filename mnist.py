import struct

import numpy as np

import datasets
from datasets.tasks import ImageClassification


_CITATION = """\
@article{lecun2010mnist,
  title={MNIST handwritten digit database},
  author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
  journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
  volume={2},
  year={2010}
}
"""

_DESCRIPTION = """\
The MNIST dataset consists of 70,000 28x28 black-and-white images in 10 classes (one for each digits), with 7,000
images per class. There are 60,000 training images and 10,000 test images.
"""
_URL = "https://huggingface.co/datasets/AnaChikashua/handwriting/resolve/main/handwriting_dataset.zip"
_NAMES = ['ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ', 'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ', 'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ', 'ხ', 'ჯ', 'ჰ']


class MNIST(datasets.GeneratorBasedBuilder):
    """MNIST Data Set"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="data",
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            citation=_CITATION,
            task_templates=[
                ImageClassification(
                    image_column="image",
                    label_column="label",
                )
            ],
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
        """This function returns the examples in the raw form."""
        for idx, filepath, image in enumerate(images):
            # extract the text from the filename
            text = [c for c in str(filepath) if not 0 <= ord(c) <= 127][0]
            yield idx, {
                "label": text,
                "image": {"path": filepath, "bytes": image.read()}
            }
