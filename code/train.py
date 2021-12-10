import logging
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
import os
from datasets import wikiHowDataset
from network.lstm import LSTMRegressor, LSTMClassifier, LSTMMulti
from network.bert import BERT_multitask_lstm_fc
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss, CrossEntropyLoss
from utils import print_header, save_model, get_lr
from meter import AverageMeter
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from scipy.stats import spearmanr


def train(args, device):
    # Classification / Ranking
    sub_task = args['subtask']
    assert sub_task in ["classification", "ranking", "multi"]

    # create Dataset, dataloader
    trainData = wikiHowDataset(args, phase="train")
    trainDataloader = DataLoader(dataset=trainData,
                                 batch_size=args["bs"],
                                 shuffle=True,
                                 pin_memory=True if torch.cuda.is_available() else False,
                                 num_workers=8)
    valData = wikiHowDataset(args, phase="dev")
    valDataloader = DataLoader(dataset=valData,
                               batch_size=args["bs"],
                               shuffle=False,
                               pin_memory=True if torch.cuda.is_available() else False,
                               num_workers=8)
    logging.info(f"Train size: {len(trainDataloader)}, Dev size: {len(valDataloader)}")
    model = None

    # TODO: other network arch
    if args["basenet"] == "lstm":
        model = LSTMMulti(input_dim=args["glove_embedding_size"],
                          embedding_dim=args["glove_embedding_size"],
                          *args["lstm_multi"])
    else:
        model = BERT_multitask_lstm_fc(basenet=args["basenet"])

    # TODO: resume training
    start_epoch = 0

    model.to(device)
    save_dir = os.path.join('model_weights', f"{args['basenet']}")
    writer = SummaryWriter(log_dir=save_dir)

    # start
    print_header("Training starts")
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])

    for ep in range(start_epoch, start_epoch + args["n_epochs"]):
        train_loss = epoch_train(model, trainDataloader, optimizer, device, args["basenet"])
        val_loss, metrics = epoch_val(model, valDataloader, device, args["basenet"])

        writer.add_scalar("Loss/train", train_loss, ep)
        writer.add_scalar("Loss/val", val_loss, ep)
        writer.add_scalar("Classification_acc/val", metrics[0], ep)
        writer.add_scalar("Classification_f1_0/val", metrics[1][0], ep)
        writer.add_scalar("Classification_f1_1/val", metrics[1][1], ep)
        writer.add_scalar("Classification_f1_2/val", metrics[1][2], ep)
        writer.add_scalar("Score_mse/val", metrics[2], ep)
        writer.add_scalar("Score_spearm/val", metrics[3], ep)

        save_path = os.path.join(save_dir, f"epoch_{ep}" + ".pth")
        save_model(model, optimizer, opt=None, epoch=ep, save_file=save_path)
        model.to(device)

        logging.info(f"Ep: {ep}\t Lr: {get_lr(optimizer):.6f}\t Train Loss: {train_loss:.4f}\t "
                     f"Val Loss: {val_loss:.4f}\t "
                     f"Val: Classification(acc/f1_0): {metrics[0]:.4f}/{metrics[1][0]}, "
                     f"Val: Classification(acc/f1_1): {metrics[0]:.4f}/{metrics[1][1]}, "
                     f"Val: Classification(acc/f1_2): {metrics[0]:.4f}/{metrics[1][2]}, "
                     f"Score(mse/spearm): {metrics[2]:.4f}/{metrics[3]:.4f}")

    print_header("Testing on the Validation set")
    val_loss, metrics = epoch_val(model, valDataloader, device)
    logging.info(f"Val Loss: {val_loss:.4f}\t "
                 f"Val: Classification(acc/f1_0): {metrics[0]:.4f}/{metrics[1][0]}, "
                 f"Val: Classification(acc/f1_1): {metrics[0]:.4f}/{metrics[1][1]}, "
                 f"Val: Classification(acc/f1_2): {metrics[0]:.4f}/{metrics[1][2]}, "
                 f"Score(mse/spearm): {metrics[2]:.4f}/{metrics[3]:.4f}")


def concat_output(output, device):
    temp = torch.tensor([]).to(device)

    temp = torch.argmax(output[:, 0:3], dim=1).view(-1, 1)
    temp = torch.cat((temp, output[:, 3].view(-1, 1)), dim=1)

    return temp


def loss_func(output, target, device):
    target = target.float()

    ce_loss = CrossEntropyLoss()
    mse_loss = MSELoss()

    loss_classification = ce_loss(output[:, 0:3], target[:, 0].long())
    loss_regression = mse_loss(output[:, 3], target[:, 1])
    loss = loss_classification + loss_regression

    # label output
    temp = concat_output(output, device)
    temp = temp.detach()

    return loss, temp


def epoch_train(model, dataloader, optimizer, device, basenet):
    model.train()
    train_loss = AverageMeter()

    # TODO: LSTM
    if basenet != "lstm":
        for batch_id, batch_data in enumerate(dataloader):
            data = batch_data[0]
            target = batch_data[1].to(device)

            token_id = data[0].to(device).squeeze()
            mask_id = data[1].to(device).squeeze()
            segment_id = data[2].to(device).squeeze()

            output = model(token_id, mask_id, segment_id)

            loss, _ = loss_func(output, target, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), target.shape[0])
    return train_loss.avg


def epoch_val(model, dataloader, device, basenet):
    model.eval()
    val_loss = AverageMeter()
    outputs = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)

    with torch.no_grad():
        # TODO: LSTM
        if basenet != "lstm":
            for batch_id, batch_data in enumerate(dataloader):
                data = batch_data[0]
                target = batch_data[1].to(device)

                token_id = data[0].to(device).squeeze()
                mask_id = data[1].to(device).squeeze()
                segment_id = data[2].to(device).squeeze()

                output = model(token_id, mask_id, segment_id)

                loss, temp = loss_func(output, target, device)
                outputs = torch.cat((outputs, temp), 0)
                targets = torch.cat((targets, target), 0)

                val_loss.update(loss.item(), target.shape[0])
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    label_acc = accuracy_score(y_true=targets[:, 0], y_pred=outputs[:, 0])
    # multi labels f-score
    label_fscore = f1_score(y_true=targets[:, 0], y_pred=outputs[:, 0], average=None)
    score_mse = mean_squared_error(y_true=targets[:, 1], y_pred=outputs[:, 1], squared=False)
    score_spearman, _ = spearmanr(targets[:, 1], outputs[:, 1])

    return val_loss.avg, [label_acc, label_fscore, score_mse, score_spearman]


if __name__ == "__main__":
    import json
    with open("config.json", "r") as f:
        conf = json.load(f)
    train(conf)
