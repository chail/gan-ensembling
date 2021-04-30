import torch


def accuracy(output, target, topk=(1,), return_preds=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # modified from : https://github.com/pytorch/examples/blob/ad775ace1b9db09146cdd0724ce9195f7f863fff/imagenet/main.py
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        if return_preds:
            preds = []
            for k in topk:
                preds.append(pred[:k])
        else:
            preds = None

        return res, preds
