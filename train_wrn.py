import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import torchvision
from torchvision import transforms
from torchvision import datasets
from data.cifar import Cifar

from models.wrn import Wide_ResNet, conv_init
from utility.eval import evaluate
from utility.initialize import initialize

from tqdm import tqdm
import argparse

def main(args):
    train_trans = [transforms.RandomHorizontalFlip(),
                   transforms.RandomRotation(15),
                   transforms.RandomCrop(32, padding=4),
                   transforms.ToTensor(),
                   transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]],
                                        std=[n/255. for n in [68.2,  65.4,  70.4]])]

    val_trans = [transforms.ToTensor(),
                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]],
                                       std=[n/255. for n in [68.2,  65.4,  70.4]])]

    train_dataset = datasets.CIFAR100('./data',
                                      train=True,
                                      transform=transforms.Compose(train_trans),
                                      download=True)

    val_dataset = datasets.CIFAR100('./data',
                                    train=False,
                                    transform=transforms.Compose(val_trans),
                                    download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # dataset = Cifar(args.batch_size, args.threads)
    # initialize(args, seed=42)

    # model = Wide_ResNet
    class WRN:
        D = args.depth
        k = args.width_factor
        droprate = args.droprate
        num_classes = 100 # CIFIAR-100

    args.logs + '_{}'.format(str(WRN.D)) + '_{}'.format(str(WRN.k))
    
    model = Wide_ResNet(WRN.D, WRN.k, WRN.droprate, WRN.num_classes)
    if args.init:
        print("CONV INIT")
        model.apply(conv_init)
    model.cuda()

    loss_CE = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.wdecay)

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.wdecay)

    decay_epoch = [60, 120, 160]
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.2)

    best_acc = 0
    os.makedirs(args.logs, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logs)
    # global_step = 0

    for epoch in range(1, args.epoch+1):
        for step, [imgs, labels] in tqdm(enumerate(train_loader)):
            # global_step += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            labels = labels.cuda()

            preds = model(imgs)
            loss = loss_CE(preds, labels)

            loss.backward()
            optimizer.step()

        step_lr_scheduler.step()
        
        train_loss = loss.item()

        model.eval()
        top1_acc, top5_acc = evaluate(model, val_loader)
        model.train()

        print("epoch : ", epoch)
        print("loss : ", train_loss)
        print("TOP 1 : ", top1_acc)
        print("TOP 5 : ", top5_acc)

        writer.add_scalar('train/Loss', train_loss, epoch)
        writer.add_scalar('val/top1', top1_acc, epoch)
        writer.add_scalar('val/top5', top5_acc, epoch)

        if best_acc < top1_acc:
            best_acc = top1_acc
            save_path = os.path.join(args.logs, args.model+'_best.pth')
            torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--width_factor", default=4, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument('-droprate', '--droprate', type=float, default=0.3)

    parser.add_argument('-batch_size', '--batch_size', type=int, default=128)
    parser.add_argument('-epoch', '--epoch', type=int, default=200)
    parser.add_argument('-lr', '--lr', type=int, default=0.1)
    parser.add_argument('-wdecay', '--wdecay', type=int, default=5e-4)
    parser.add_argument('-momentum', '--momentum', type=int, default=0.9)
    parser.add_argument('-optim', '--optim', type=str, default='sgd', help='sgd or adam')

    parser.add_argument('-logs', '--logs', type=str, default='logs')
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='# of gpu device you want to use')
    parser.add_argument("--threads", default=0, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument('-init', '--init', action='store_true', default=False, help='use conv_init')

    args = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.logs = os.path.join(args.logs, now)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    cudnn.benchmark = True
    main(args)