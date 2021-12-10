import logging
import pandas as pd


def evaluation(args):
    sub_task = args['subtask']
    path_dev = args['path_to_dev']
    path_dev_labels = args['path_to_dev_labels'][sub_task]

    # load dev set
    logging.debug(f"Read dev dataset from file {path_dev}")
    dev_df = pd.read_csv(path_dev, sep="\t", quoting=3)
    dev_ids, dev_instances = retrieve_instances_from_dataset(dev_df)


if __name__ == "__main__":
    import json
    with open("config.json", "r") as f:
        conf = json.load(f)
    evaluation(conf)
