import logging
from typing import List, Tuple
import os
import pickle
import pandas as pd
import numpy as np


def load_glove_map(glove_path):
    """Load GloVe mapping from txt or cached pickle file. For unseen words, use
    random normal vectors.

    :param: str path to glove txt file
    :return: dict of (word, embedding)
    """

    glove_pkl_path = glove_path[:-4] + ".pkl"

    # load from pickle file
    if os.path.exists(glove_pkl_path):
        logging.debug(f"GloVe map found at {glove_pkl_path}, load from cache.")
        with open(glove_pkl_path, "rb") as f:
            glove_map = pickle.load(f)
    else:
        logging.debug(f"Creating GloVe map from {glove_path}...")
        df = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
        glove_map = {key: val.values for key, val in df.T.items()}
        # dump pickle
        with open(glove_pkl_path, "wb") as f:
            pickle.dump(glove_map, f)
    return glove_map


def get_glove_embedding(glove_map, x, emb_dim):
    """Convert word to GloVe embedding

    :param glove_map: dict of glove embedding
    :param x: str single word
    :param emb_dim: glove dim size
    :return: np.arrays (len, emb_dim) glove embedding matrix
    """
    matrix = np.zeros(shape=(len(x), emb_dim))

    for idx, word in enumerate(x):
        if word in glove_map.keys():
            matrix[idx] = glove_map[word]
        else:
            # random vector if no corresponding embedding
            matrix[idx] = np.random.normal(scale=0.6, size=(emb_dim,))
    return matrix


def convert_instances_to_glove_embedding(args, train_instances):
    """Convert instances to GloVe embedding matrix list

    :param args: dict of configurations
    :param train_instances: list of inputs sentences
    :return list of gloves embedding matrix with shape of (len, emb_size)
    """

    logging.debug("Loading GloVe mapping...")
    glove_map = load_glove_map(args["glove_path"])
    # Convert text to GloVe embeddings
    logging.debug("Converting training set into GloVe Embeddings...")
    glove_map = load_glove_map(args['glove_path'])
    glove_dim_size = args["glove_embedding_size"]
    train_gloves = list(map(lambda x: get_glove_embedding(glove_map, x, glove_dim_size), train_instances))
    return train_gloves
