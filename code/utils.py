import logging
from typing import List, Tuple
import pandas as pd
import torch


def print_header(txt):
    """Print any string with specific format
    """
    print("\n")
    print("=" * 80)
    print(" " * 30, txt)
    print("=" * 80)


def load_weights(model, resume_model=None):
    """
        Load saved model state dictionary
        Input:
            resume_model : path of the saved checkpoint to be loaded
        Output:
            model        : weight loaded model
            start_epoch  : starting epoch of the training
    """
    logging.debug(f"=> Loading model weights from {resume_model}")
    ckpt = torch.load(resume_model, map_location=torch.device('cpu'))

    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt['model']

    state_dict_pretrained = state_dict
    # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    model_dict = model.state_dict()

    state_dict = {k: v for k, v in state_dict_pretrained.items() if
                  (k in model_dict) and (v.shape == model_dict[k].shape)}

    # overwrite entries in the existing state dict
    model_dict.update(state_dict)

    # load the new state dict
    model.load_state_dict(model_dict)

    print("Length of model-dict  : {}".format(len(model_dict)))
    print("Length of loaded dict : {}".format(len(state_dict)))

    if len(model_dict) != len(state_dict):
        not_in_state_dict = {k: v.shape for k, v in model_dict.items() if k not in state_dict}
        print("Layers which are in model-dict but not in loaded state-dict:")
        print(not_in_state_dict)

        not_in_model_dict = {k: v.shape for k, v in state_dict_pretrained.items() if (k not in state_dict)}
        print("Layers which are in loaded state-dict but not in model-dict:")
        print(not_in_model_dict)

    start_epoch = ckpt['epoch'] + 1
    # start_epoch = 1
    return model, start_epoch


def save_model(model, optimizer, opt, epoch, save_file):
    if isinstance(model, torch.nn.DataParallel):
        state = {
            'opt': opt,
            'model': model.cpu().module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    else:
        state = {
            'opt': opt,
            'model': model.cpu().state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    torch.save(state, save_file)


def get_lr(optimizer):
    """
        Returns the learning rate for printing
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def write_predictions_to_file(
        path_to_predictions: str, ids: List[str], predictions: List, subtask: str
) -> pd.DataFrame:
    """Write the instance indices and predictions to a tsv file.

    :param path_to_predictions: str path to file where to write the predictions
    :param ids: list of str instance indices
    :param predictions: list of predictions
    :param subtask: str indicating "ranking" or "classification"
    :return: pandas dataframe with ids and predictions
    """
    if subtask == "classification":
        predictions = convert_class_indices_to_labels(predictions)

    dataframe = pd.DataFrame({"Id": ids, "Label": predictions})
    logging.info(f"--> Writing predictions to {path_to_predictions}")
    dataframe.to_csv(path_to_predictions, sep="\t", index=False, header=False)

    return dataframe


def convert_class_indices_to_labels(class_indices: List[int]) -> List[str]:
    """Convert integer class indices to str labels.

    :param class_indices: list of int class indices (0 to 2)
    :return: list of label strs from set "IMPLAUSIBLE" / "NEUTRAL" / "PLAUSIBLE"
    """
    labels = ["IMPLAUSIBLE", "NEUTRAL", "PLAUSIBLE"]
    return [labels[class_index] for class_index in class_indices]
