# https://github.com/davda54/sam
import os
import datetime

import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets
from utility.eval import evaluate

# from model.wide_res_net import WideResNet
from models.wrn import Wide_ResNet, conv_init
from models.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from torch.utils.tensorboard import SummaryWriter
import sys; sys.path.append("..")
from sam import SAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.5, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=4, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='# of gpu device you want to use')
    parser.add_argument('-logs', '--logs', type=str, default='logs_sam')
    args = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.logs = os.path.join(args.logs, now)

    initialize(args, seed=42)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # change allocation of current GPU

    val_trans = [transforms.ToTensor(),
                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]],
                                       std=[n/255. for n in [68.2,  65.4,  70.4]])]

    val_dataset = datasets.CIFAR100('./data',
                                    train=False,
                                    transform=transforms.Compose(val_trans),
                                    download=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = Wide_ResNet(args.depth, args.width_factor, args.dropout, num_classes=100)
    # model.apply(conv_init)
    model.to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    loss_CE = nn.CrossEntropyLoss

    best_acc = 0
    os.makedirs(args.logs, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logs)
    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            smooth_crossentropy(model(inputs), targets).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)


        model.eval()
        log.eval(len_dataset=len(dataset.test))
        top1_acc, top5_acc = evaluate(model, val_loader, verbose=False)

        # train_loss = loss_CE(inputs, targets).item()
        writer.add_scalar('train/Loss', loss.mean().item(), epoch)
        writer.add_scalar('val/top1', top1_acc, epoch)
        writer.add_scalar('val/top5', top5_acc, epoch)

        if best_acc < top1_acc:
            best_acc = top1_acc
            save_path = os.path.join(args.logs, str(args.depth) +'_' + str(args.width_factor) + '_best.pth')
            torch.save(model.state_dict(), save_path)

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()