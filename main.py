import argparse
import os
import time

import torch
from torch.optim.adam import Adam

import utils
from model.data_loader import fetch_dataloader
from model.net import VGG
from train import evaluate, train_and_eval

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='dataset/flowers',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='config',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")
parser.add_argument('--mode', default='train', help="specific run train or evaluate")
def accuracy(outputs, targets):
    outputs = torch.argmax(outputs, dim = 1).cpu().numpy()
    # print(outputs)
    targets = targets.cpu().numpy()
    return len (outputs[outputs == targets]) / len(targets)

if __name__ == "__main__":
    # load params
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)
    utils.set_logger(os.path.join(args.model_dir, "log/" + args.mode + "{}".format(time.time()) + ".log"))

    model = VGG((128, 128), 5)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    dataloaders = fetch_dataloader(args.data_dir, [0.6, 0.2, 0.2], params)
    if(args.restore_file):
        model.load_state_dict(torch.load(args.restore_file))
    if(torch.cuda.is_available()):
        model = model.cuda()
    if(args.mode == 'train'):
        train_and_eval(model, loss_fn, dataloaders['train'], dataloaders['val'], optimizer, params.epoch, accuracy, os.path.join(args.model_dir, "model"))
    elif (args.restore_file):
        evaluate(model, loss_fn, dataloaders['test'], accuracy)
