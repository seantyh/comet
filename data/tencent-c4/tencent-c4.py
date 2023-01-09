from pathlib import Path
import pickle
import datasets


_DESCRIPTION = "Tencent Chinese NN compounds embedding dataset"


class TencentC4(datasets.GeneratorBasedBuilder): #type: ignore
    """Tencent C4 dataset."""

    def _info(self):
        return datasets.DatasetInfo(  #type: ignore
            description=_DESCRIPTION,
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
        data_file = dl_manager.download("../tencent-compound-c2.pkl")
        
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