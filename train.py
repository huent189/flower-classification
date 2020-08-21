import logging
import time

import torch
from torch.autograd import Variable


def train(model, optimizer, loss_fcn, data, accumulated_metric):
    t0 = time.time()
    model.train()
    metric = 0
    for X, y in data:
        if(torch.cuda.is_available()):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
        X, y = Variable(X), Variable(y)
        y_hat = model(X)
        loss = loss_fcn(y_hat, y)
        metric = accumulated_metric(y_hat, y, 5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    metric = metric / len(data) * 100
    logging.info("Train: Time: {:.2f}| Loss: {:.2f}|Accuracy: {:.2f}".format(time.time() - t0, loss, metric))
    return loss.item(), metric

def evaluate(model, loss_fcn, data, accumulated_metric):
    t0 = time.time()
    model.eval()
    metric = 0
    for X, y in data:
        if(torch.cuda.is_available()):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
        X, y = Variable(X), Variable(y)
        y_hat = model(X)
        loss = loss_fcn(y_hat, y)
        metric += accumulated_metric(y_hat, y, 5)
    metric = metric / len(data) * 100
    print("Eval\t\t\t\t\tTime: {:.2f}| Loss: {:.2f}|Accuracy: {:.2f}".format(time.time() - t0, loss, metric))
    return loss.item(), metric

def train_and_eval(model, loss_fn, train_dataloader, val_dataloader, optimizer, total_epoch, accumulated_metric, save_path):
    best_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for i in range(total_epoch):
        print("Epoch {}:".format(i))
        train_loss, train_acc = train(model, optimizer, loss_fn, train_dataloader, accumulated_metric)
        val_loss, val_acc = evaluate(model, loss_fn, val_dataloader, accumulated_metric)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if val_acc > best_acc:
            torch.save(model.state_dict(), save_path + "best.pth")
            best_acc = val_acc
    return train_losses, train_accs, val_losses, val_accs
