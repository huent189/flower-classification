import torch


def confusion_matrix(y_true, y_pred, num_classes=None):
    y_true = torch.argmax(y_true, dim=1)
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    classes = torch.ones_like(y_true)
    cf_mx = torch.sparse.LongTensor(torch.stack([y_true, y_pred]), classes, torch.Size([num_classes, num_classes])).to_dense()
    return cf_mx


def accuracy(outputs, targets, num_classes=None):
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    # print(outputs)
    targets = targets.cpu().numpy()
    return len(outputs[outputs == targets]) / len(targets)
