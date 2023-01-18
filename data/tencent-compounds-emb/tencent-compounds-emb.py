from pathlib import Path
import pickle
import datasets
import numpy as np

_DESCRIPTION = "Tencent Chinese NN compounds embedding dataset - {name}"
_VERSION = datasets.Version("0.1.0")

class TencentCompound(datasets.GeneratorBasedBuilder): #type: ignore
    """Tencent C4 dataset."""

    BUILDER_CONFIG = [
        datasets.BuilderConfig(  #type: ignore
            name="tencent-c4", version=_VERSION, 
            description="Tencent Compound C4 dataset"),
        datasets.BuilderConfig(  #type: ignore
            name="tencent-c2", version=_VERSION,
            description="Tencent Compound C2 dataset")
    ]

    DEFAULT_CONFIG_NAME="tencent-c4"
    
    def _info(self):
        return datasets.DatasetInfo(  #type: ignore
            description=_DESCRIPTION.format(name=self.config.name),
            features=datasets.Features(
                {
                    "compound": datasets.Sequence(datasets.Value("float32")),
                    "consts": datasets.Sequence(datasets.Value("float32")),
                    "compound_text": datasets.Value("string"),
                    "const1_text": datasets.Value("string"),
                    "const2_text": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/seantyh/comet")

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.name == "tencent-c4":
            data_file = dl_manager.download("tencent-compound-c4.pkl")
        elif self.config.name == "tencent-c2":
            data_file = dl_manager.download("tencent-compound-c2.pkl")
        else:
            raise ValueError("Unknown dataset name: {}".format(self.config.name))
        
        data = pickle.loads(Path(data_file).read_bytes()) #type: ignore

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"data_split": data["train"]}),  # type: ignore
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"data_split": data["val"]}),  # type: ignore
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"data_split": data["test"]}),  # type: ignore
        ]

    def _generate_examples(self, data_split):
        
        for idx in range(len(data_split["comps"])):
            yield idx, {
                "compound": data_split["comps"][idx],
                "consts": data_split["consts"][idx],
                "compound_text": data_split["comps_text"].iloc[idx],
                "const1_text": data_split["c1_text"].iloc[idx],
                "const2_text": data_split["c2_text"].iloc[idx]
            }