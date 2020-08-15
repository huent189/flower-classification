import torch
def confusion_matrix(y_true, y_pred, num_classes):
    classes = torch.ones(num_classes)
    cf_mx = torch.sparse.LongTensor([y_true, y_pred], classes, torch.Size([classes, classes])).to_dense()
    return cf_mx

