import string
import pandas as pd
import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, DebertaTokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from typing import List
from preprocessing import load_glove_map, get_glove_embedding
import logging


class wikiHowDataset(Dataset):
    """wikiHowDataset with BERT token
    """

    def __init__(self,
                 args,
                 max_length=128,
                 stop_words=False,
                 phase="train"):
        super(wikiHowDataset, self).__init__()

        self.task = args["subtask"]
        assert self.task in ["classification", "ranking", "multi"]

        assert phase in ["train", "dev"]

        # get instance path
        self.is_test = args["phase"] == "test"

        instance_path = args["path_to_test"] if self.is_test else args["path_to_" + phase]

        if stop_words:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = None
        # load instances
        self.data = self.retrieve_instances_from_dataset(instance_path)

        if not self.is_test:
            ranking_path = args["path_to_" + phase + "_labels"]["ranking"]
            classification_path = args["path_to_" + phase + "_labels"]["classification"]
            if self.task == "ranking" or self.task == "multi":
                self.ranking_labels = self.retrieve_labels_from_dataset_for_ranking(ranking_path)
            if self.task == "classification" or self.task == "multi":
                self.classification_labels = self.retrieve_labels_from_dataset_for_classification(classification_path)

        self.basenet = args["basenet"]

        if self.basenet == 'bert':
            logging.info("Tokenizer: bert-base-uncased")
            self.token = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.basenet == 'ernie':
            logging.info("Tokenizer: ernie-2.0-en")
            self.token = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
        elif self.basenet == 'roberta':
            logging.info("Tokenizer: roberta-base")
            self.token = RobertaTokenizer.from_pretrained('roberta-base')
        elif self.basenet == 'deberta':
            logging.info("Tokenizer: microsoft/deberta-base")
            self.token = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        elif self.basenet == 'glove':
            logging.info("Tokenizer: GloVe")
            self.emb_size = args["glove_embedding_size"]
            self.token = load_glove_map(args["glove_path"])

        self.max_length = max_length
        self.segment_id = torch.tensor([1] * self.max_length).view(1, -1)

    def text_cleaning(self,
                      text: str,
                      lemma=True) -> str:
        """Text cleansing

        Ref: https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing

        * lower case
        * remove multiple spaces
        * remove stop words
        * lemmatize
        :param text: input string
        :param stop_word: boolean, whether remove the stop words
        :param lemma: boolean, whether lemmatize the text
        :return text: cleaned text
        """
        # lower
        text = text.lower()

        # multiple spaces
        text = text.replace(r"\s+", ' ')

        # punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # stop words
        if self.stop_words is not None:
            text = " ".join([w for w in text.split() if w not in self.stop_words])

        # LEMMA
        if lemma:
            lemmatizer = WordNetLemmatizer()
            text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])

        return text

    def retrieve_instances_from_dataset(self, filename: str) -> pd.DataFrame:
        """Retrieve sentences with insertions from dataset.

        :param filename: path of the tsv file
        :return: dataframe with id and instances columns
        """
        logging.debug(f"Read instances from file {filename}")
        df = pd.read_csv(filename, sep="\t", quoting=3)
        df = df.fillna("")

        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        ids = []
        instances = []

        # TODO: Previous + sentence + follow-up
        for _, row in df.iterrows():
            # fill the blank with the fillers
            # generate id formatted as: "DataID_FillerID" (str)
            for filler_index in range(1, 6):
                ids.append(f"{row['Id']}_{filler_index}")

                # remove separator
                previous_txt = row["Previous context"].replace("(...)", "")
                follow_txt = row["Follow-up context"].replace("(...)", "")

                sent_with_filler = row["Sentence"].replace(
                    "______", row[f"Filler{filler_index}"]
                )
                previous_txt = self.text_cleaning(previous_txt)
                follow_txt = self.text_cleaning(follow_txt)
                sent_with_filler = self.text_cleaning(sent_with_filler)

                instances.append(previous_txt + sent_with_filler + follow_txt)

        output = pd.DataFrame({"id": ids, "instances": instances})

        return output

    def retrieve_labels_from_dataset_for_ranking(self, filename: str) -> List[float]:
        """Retrieve ranking labels from dataset.

        DO NOT MESS THE ORDER!

        :param filename: file path string
        :return: list of rating floats
        """
        logging.debug(f"Read ranking labels from {filename}")
        labels = pd.read_csv(filename, sep="\t", header=None, names=["Id", "Label"])
        # the labels are already in the right order for the training instances
        return list(labels["Label"])

    def retrieve_labels_from_dataset_for_classification(self, filename: str) -> List[int]:
        """Retrieve classification labels from dataset.

        DO NOT MESS THE ORDER!

        :param filename: file path string
        :return: list of int class labels 0, 1 or 2 (IMPLAUSIBLE, NEUTRAL, PLAUSIBLE)
        """
        logging.debug(f"Read classification labels from {filename}")
        labels = pd.read_csv(filename, sep="\t", header=None, names=["Id", "Label"])

        label_strs = list(labels["Label"])
        label_ints = []

        for label_str in label_strs:
            if label_str == "IMPLAUSIBLE":
                label_ints.append(0)
            elif label_str == "NEUTRAL":
                label_ints.append(1)
            elif label_str == "PLAUSIBLE":
                label_ints.append(2)
            else:
                raise ValueError(f"Label {label_str} is not a valid plausibility class.")
        # the labels are already in the right order for the training instances
        return label_ints

    def get_tokenized_text(self, text):
        # marked_text = "[CLS] " + text + " [SEP]"
        encoded = self.token(text=text,  # the sentence to be encoded
                             add_special_tokens=True,  # add [CLS] and [SEP]
                             max_length=self.max_length,  # maximum length of a sentence
                             padding='max_length',  # add [PAD]s
                             return_attention_mask=True,  # generate the attention mask
                             return_tensors='pt',  # return PyTorch tensors
                             truncation=True
                             )

        input_id = encoded['input_ids']
        mask_id = encoded['attention_mask']

        return input_id, mask_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text = self.data.iloc[idx]['instances']

        label = []

        if not self.is_test:
            if self.task == "multi":
                label.append(self.classification_labels[idx])
                label.append(self.ranking_labels[idx])
            elif self.task == "classification":
                label.append(self.classification_labels[idx])
            elif self.task == "ranking":
                label.append(self.ranking_labels[idx])
        else:
            label.append(self.data.iloc[idx]['id'])

        label = torch.tensor(label)

        if self.basenet == "glove":
            emb = get_glove_embedding(self.token, text, self.emb_size)
            return torch.tensor(emb), label
        else:
            input_id, mask_id = self.get_tokenized_text(text)
            return [input_id, mask_id, self.segment_id], label


if __name__ == "__main__":
    import json
    with open("config.json", "r") as f:
        conf = json.load(f)
    ds = wikiHowDataset(conf)


