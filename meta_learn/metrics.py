import torch
import torch.nn.functional as F

from sklearn import metrics as sk_metrics


def _flatten(tensor):
    if len(tensor.shape) == 2:
        tensor = tensor.argmax(1)
    return tensor


def _accuracy(preds, targs):
    if len(targs.shape) == 2:
        targs = targs.max(1)[1]
    preds = preds.max(1)[1].type_as(targs)
    correct = preds.eq(targs).double()
    correct = correct.sum()
    return correct / len(targs)


def accuracy(preds, targs):
    preds = _flatten(preds).cpu().numpy()
    targs = _flatten(targs).cpu().numpy()
    return sk_metrics.accuracy_score(preds, targs)


def f1_score(preds, targs, average='micro'):
    preds = _flatten(preds).cpu().numpy()
    targs = _flatten(targs).cpu().numpy()
    return sk_metrics.f1_score(preds, targs, average=average)


def recall(preds, targs, threshold=0.5):
    pred_pos = preds > threshold
    true_pos = (pred_pos[pred_pos] == targs[pred_pos].byte()).sum()
    return true_pos/targs.float().sum()


def macro_recall(preds, targs):
    if len(targs.shape) == 2:
        targs = targs.argmax(1)
    if len(preds.shape) == 2:
        preds = preds.argmax(1)

    n_classes = targs.max()


def precision(preds, targs, threshold=0.5):
    pred_pos = preds > threshold
    true_pos = (pred_pos[pred_pos] == targs[pred_pos].byte()).sum()
    return true_pos/pred_pos.float().sum()


def fbeta_score(preds, targs, beta=1., threshold=0.5, average='macro'):
    preds = torch.sigmoid(preds)
    prec = precision(preds, targs, threshold=threshold)
    rec = recall(preds, targs, threshold=threshold)
    beta2 = beta * beta
    num = (1 + beta2) * prec * rec
    denom = beta2 * (prec + rec)

    if denom == 0:
        return 0.

    return num/denom

def cross_entropy_loss(pred_logits, target, weighted=False):
    """
    Calculates cross-entropy between a tensor of logits and
    a given target probability distribution.
    This is used when the target is a probability distribution
    as torch.nn.functional.cross_entropy expects the target label
    as a scalar.
    """
    log_probs = -F.log_softmax(pred_logits, dim=1)
    if weighted:
        log_probs = log_probs * F.softmax(1/target.sum(0), 0)
    loss = (log_probs * target).sum(dim=1)
    loss = loss.mean()
    return loss


def focal_loss(pred_logits, target, gamma):
    """
    Calculates focal loss between a tensor of logits and
    a given target probability distribution.
    This is used when the target is a probability distribution
    as torch.nn.functional.cross_entropy expects the target label
    as a scalar.
    """
    log_probs = -F.log_softmax(pred_logits, dim=1)
    loss = (torch.pow(1-torch.exp(-log_probs), gamma)
            * log_probs * target).sum(dim=1)
    loss = loss.mean()
    return loss


if __name__ == "__main__":
    y_true = torch.LongTensor([0, 1, 2, 0, 1])
    y_pred = torch.LongTensor([0, 2, 1, 0, 0])
    assert accuracy(y_true, y_pred)==0.4
