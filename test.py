import torch
from torch.utils.data import DataLoader

# import torchvision
from torchvision import transforms
from torchvision import datasets

from models.wrn import Wide_ResNet
# from models import wrn
from utility.eval import evaluate
from pathlib import Path

def print_acc(model, path, data_loader, f):
    model.load_state_dict(torch.load(str(path)))
    model.cuda()
    model.eval()
    top1_acc, top5_acc = evaluate(model, data_loader)
    
    print("-----{}-----".format(path.name))
    print("top1 : ", round(top1_acc, 2))
    print("top5 : ", round(top5_acc, 2))

    f.write("-----{}-----".format(path.name) + "\n")
    f.write("top1 : {}".format(round(top1_acc, 2)) + "\n")
    f.write("top5 : {}".format(round(top5_acc, 2)) + "\n")

def main():

    class T:
        D = 40
        k = 4
        droprate = 0.0
        num_classes = 100 # CIFIAR-100

    class S:
        D = 16
        k = 4
        droprate = 0.0
        num_classes = 100 # CIFIAR-100

    val_trans = [transforms.ToTensor(),
                 transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]],
                                      std=[n/255. for n in [68.2,  65.4,  70.4]])]

    val_dataset = datasets.CIFAR100('./data',
                                    train=False,
                                    transform=transforms.Compose(val_trans),
                                    download=True)

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)

    
    Student = Wide_ResNet(S.D, S.k, S.droprate, S.num_classes)
    Student.cuda()
    Student.eval()
    Teacher = Wide_ResNet(T.D, T.k, T.droprate, T.num_classes)
    Teacher.cuda()
    Teacher.eval()

    with open('results.txt', 'w') as f:
        for path in sorted(list(Path('./pths').rglob('*.pth'))):
            if 'teacher' in str(path).lower():
                print_acc(Teacher, path, val_loader, f)
            else:
                print_acc(Student, path, val_loader, f)

if __name__ == '__main__':
    main()