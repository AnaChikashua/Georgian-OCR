import datasets

logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {Small image-text set},
author={James Briggs},
year={2022}
}
"""
_DESCRIPTION = """
Georgian language handwriting dataset!
"""
_URL = 'https://huggingface.co/datasets/AnaChikashua/handwriting/resolve/main/handwriting_dataset.tar.gz'
class HandwritingData(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description =_DESCRIPTION,
            ciation = _CITATION,
            features = datasets.Features(
            {"alphabet": datasets.Value("string"),
             "image": datasets.Image()
             }
            ),
            supervised_keys = None,
            homepage = "https://huggingface.co/datasets/AnaChikashua/handwriting",
        )
    def _split_generators(self, dl_manager):
        path = dl_manager.dowload(_URL)
        image_iters = dl_manager.iter_archive(path)
        return [
            datasets.SplitGenerator(
            name = datasets.Split.TRAIN,
            gen_kwargs = {"images": image_iters}
            ),
        ]
    def _generate_examples(self, images):
        """This function returns the examples in the raw (text) form."""
        idx = 0
        # Iterate through images
        for filepath, image in images:
            # extract the text from the filename
            text = filepath.split("/")[-1].split(".")[0]
            yield idx, {
                "alphabet": text,
                "image": {"path": filepath, "bytes": image.read()}
            }
            idx += 1