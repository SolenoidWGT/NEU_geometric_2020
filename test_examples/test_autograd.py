import torch


if __name__ == "__main__":
    x = torch.tensor([[0,1,2,3],[4,5,6,7],[8,9,10,11]], requires_grad=True).float()
    y = torch.tensor([[0,0,0,0],[0,0,0,0],[1,1,1,1]])
    z = x.pow(2)+3*y.pow(2)     # z = x^2+3y^2, dz/dx=2x, dz/dy=6y
    z.backward()   #纯标量结果可不写占位变量
    pass