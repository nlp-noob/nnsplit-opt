from utils import evaluate
from utils.evaluate import OpenSubtitlesDataset, Evaluator
from torch.utils.data import Subset
from nnsplit import NNSplit
from rich.console import Console
from tqdm import tqdm

import spacy
import numpy as np
import torch
import pandas as pd


console = Console()


class NNSplitInterface:
    def __init__(self, splitter):
        self.splitter = splitter

    def split(self, texts):
        out = []
        console.print("Splitting the sentence...", style="green")
        for split in self.splitter.split(texts):
            out.append([str(x) for x in split])

        return out

class SpacyInterface:
    def __init__(self, name, use_sentencizer):
        if use_sentencizer:
            nlp = spacy.load(name, disable=["tagger", "parser", "ner"])
            nlp.add_pipe(nlp.create_pipe("sentencizer"))
        else:
            nlp = spacy.load(name, disable=["tagger", "ner"])
    
        self.nlp = nlp
    
    def split(self, texts):
        out = []
        for doc in self.nlp.pipe(texts):
            sentences = []
    
            for sent in doc.sents:
                sentences.append("".join([x.text + x.whitespace_ for x in sent]))
    
            out.append(sentences)
    
        return out


def main():
    data = [
        [
            "English",
            Subset(OpenSubtitlesDataset("./data/example/en.txt", 1_000_000), np.arange(100_000)),
            {
                "NNSplit": NNSplitInterface(NNSplit("models/example_model.onnx")),
            }
        ],
    ]
    eval_setups = {
        "Clean": (0.0, 0.0),
        "Partial punctuation": (0.5, 0.0),
        "Partial case": (0.0, 0.5),
        "Partial punctuation and case": (0.5, 0.5),
        "No punctuation and case": (1.0, 1.0),
    }

    results = {}
    preds = {}

    for dataset_name, dataset, targets in data:
        results[dataset_name] = {}
        for eval_name, (remove_punct_prob, lower_start_prob) in eval_setups.items():
            results[dataset_name][eval_name] = {}
            evaluator = Evaluator(dataset, remove_punct_prob, lower_start_prob, punctuation=".?!")
            for target_name, interface in targets.items():
                console.print("Evaluating in \"{}\"...".format("eval_name"), style="green")
                correct = evaluator.evaluate(interface.split)
                preds[f"{dataset_name}_{eval_name}_{target_name}"] = {
                "samples": evaluator.texts,
                "correct": correct,
                }
                results[dataset_name][eval_name][target_name] = correct.mean()

    result = pd.DataFrame.from_dict(results["English"]).T
    console.print(result)


if __name__ == "__main__":
    main()

