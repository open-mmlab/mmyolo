import torch

ckp1 = torch.load('checkpoint/damoyolo_s_mmyolo.pth')
ckp2 = torch.load('checkpoint/new_damoyolo.pth')
ckp3 = torch.load('checkpoint/damoyolo_tinynasL25_S.pth')

for keys in ckp1:
    if keys in ckp2:
        print(keys)
    else:
        print('no')