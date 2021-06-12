from tqdm import tqdm as tqdm_wrap
# from tqdm import tqdm
import torch

# https://github.com/bearpaw/pytorch-classification/blob/24f1c456f48c78133088c4eefd182ca9e6199b03/utils/eval.py#L5
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0))
    return res

@torch.no_grad()
def evaluate(model, data_loader, verbose=True):
    total_top1, total_top5 = 0.0, 0.0
    tqdm = tqdm_wrap
    if not verbose:
        tqdm = lambda x : x
    for step, [imgs, labels] in tqdm(enumerate(data_loader)):
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        preds = model(imgs)
        top1, top5 = accuracy(preds.data, labels.data, topk=(1, 5))
        total_top1 += top1
        total_top5 += top5

    top1_acc = (total_top1/len(data_loader.dataset)).item()
    top5_acc = (total_top5/len(data_loader.dataset)).item()
    return top1_acc, top5_acc