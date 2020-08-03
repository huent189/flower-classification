import logging
import time

import torch
from torch.autograd import Variable


def train(model, optimizer, loss_fcn, data):
    t0 = time.time()
    model.train()
    for X, y in data:
        if(torch.cuda.is_available()):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
        X, y = Variable(X), Variable(y)
        y_hat = model(X)
        loss = loss_fcn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info("Train\tTime: {:.2f}| Loss: {:.2f}:".format(time.time() - t0, loss))
    # logging.info("Loss: {:.2f}".format(loss))
    return loss

def evaluate(model, loss_fcn, data, accumulated_metric):
    t0 = time.time()
    model.eval()
    acc = 0
    for X, y in data:
        if(torch.cuda.is_available()):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
        X, y = Variable(X), Variable(y)
        y_hat = model(X)
        loss = loss_fcn(y_hat, y)
        acc += accumulated_metric(y_hat, y)
    acc = acc / len(data)
    logging.info("Eval\t\t\t\t\tTime: {:.2f}| Loss: {:.2f}| Accuracy: {:.2f}".format(time.time() - t0, loss, acc))
    return loss, acc

def train_and_eval(model, loss_fn, train_dataloader, val_dataloader, optimizer, total_epoch, accumulated_metric, save_path):
    best_loss = 9999
    best_idx = -1
    for i in range(total_epoch):
        logging.info("Epoch {}:".format(i))
        train(model, optimizer, loss_fn, train_dataloader)
        val_loss, _ = evaluate(model, loss_fn, val_dataloader, accumulated_metric)
        if val_loss < best_loss:
            best_idx = i
            torch.save(model.state_dict(), save_path + "train.{}th.pth".format(i))
            best_loss = val_loss
    return best_idx
