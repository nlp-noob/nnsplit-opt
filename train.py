import json

from utils.labeler import Labeler, SpacySentenceTokenizer, SpacyWordTokenizer
from pytorch_lightning.trainer import Trainer
from tqdm.auto import tqdm
from model import Network
from utils.text_data import MemoryMapDataset

from rich.console import Console
from tqdm import tqdm


console = Console()


def main():
    parser = Network.get_parser()
    hparams = parser.parse_args([])

    # you can costomize with 
    # text_dataset = MemoryMapDataset(
    #         "./data/greek_test/texts.txt", 
    #         "./data/greek_test/slices.pkl",
    #         )
    sample_text = "A few things, he does not feel that things will be successful between you. A few things, he does not feel that things will be successful between you. A few things, he does not feel that things will be successful between you. A few things, he does not feel that things will be successful between you."
    text_dataset = [
        sample_text for i in range(60000)
    ]



    labeler = Labeler(
        [
            SpacySentenceTokenizer(
                "en_core_web_sm", 
                lower_start_prob=0.7, 
                remove_end_punct_prob=0.7, 
                punctuation=".?!",
                ),
            SpacyWordTokenizer("en_core_web_sm"),
        ]
    )

    hparams.gpus = 1
    hparams.max_epochs = 4
    hparams.train_size = 5_000
    hparams.predict_indices = [0, 1] # which split levels of the labeler to predict
    # how to weigh the selected indices
    # in general sentence boundary detection should be weighed the highest
    hparams.level_weights = [0.1, 2.0]

    model = Network(
        text_dataset,
        labeler,
        hparams,
    )

    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model)
    # onnx metadata which determines how to use the prediction indices to split text
    metadata = {
        "split_sequence": json.dumps(
            {
                "instructions": [
                    ["Sentence", {"PredictionIndex": 0}],
                    ["Token", {"PredictionIndex": 1}],
                    ["_Whitespace", {"Function": "whitespace"}],
                ]
            }
        )
    }
    
    model.store("en", metadata)



if __name__ == "__main__":
    main()

