import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_path", help="The src path of the data.", type=src, default="./data/cluster_src/splitted_old.json")

    args = parser.parse_args()
    jf = open(args.src_data_path, "r")
    data = json.load(jf)
    jf.close()

    for star_id, cluster_items in data.items():
        for cluster_item in cluster_items:
            cluster_list = cluster



if __name__ == "__main__":
    main()
