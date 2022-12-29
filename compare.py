import torch

ckp1 = torch.load('checkpoint/damoyolo_m_mmyolo.pth')
ckp2 = torch.load('checkpoint/damoyolo_m.pth')
ckp3 = torch.load('checkpoint/damoyolo_tinynasL35_M.pth')

for keys in ckp1:
    if keys in ckp2:
        continue
    else:
        print('no')