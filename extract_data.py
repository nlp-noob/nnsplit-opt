import json
import sys
import argparse

from tqdm import tqdm
from rich.console import Console

from utils.text_data import MemoryMapDataset, xml_dump_iter


console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="The path to the .xml.bz2", type=str, default="./data/greek_test/elwiki-20181001-corpus.xml")
    parser.add_argument("--extract_out_dir", help="The path to the extract out.", type=str, default="./data/greek_test/")
    args = parser.parse_args()

    xml_iter = xml_dump_iter(
        args.data_path,
        min_text_length=300, 
        max_text_length=5000,
        )
    next(xml_iter)

    MemoryMapDataset.iterator_to_text_and_slices(
        xml_iter, 
        args.extract_out_dir + "texts.txt", 
        args.extract_out_dir + "slices.pkl",
        max_n_texts=10_000_000
        )




if __name__ == "__main__":
    main()

