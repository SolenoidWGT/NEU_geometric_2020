import torch
import os
import importlib
import os.path as osp
from my_scatter import scatter_sum
torch.ops.load_library("../my_scatter/_scatter.so")

print(torch.ops.my_scatter.scatter_sum)
# print(torch.ops.my_scatter.scatter_add_)
# print(scatter_max)

print(torch.Tensor.scatter_add_)
print(scatter_sum)
# print(torch.ops.torch_scatter.scatter_min)

edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [1, 2, 3, 0, 3, 0, 3, 0, 1, 2]]).cuda()
x = torch.tensor([[0., 4., 0., 4., 3., 0., 1., 2.],
        [0., 4., 0., 4., 3., 0., 1., 2.],
        [0., 4., 0., 4., 3., 0., 1., 2.],
        [2., 3., 4., 3., 3., 2., 4., 0.],
        [2., 3., 4., 3., 3., 2., 4., 0.],
        [3., 3., 1., 2., 1., 2., 4., 3.],
        [3., 3., 1., 2., 1., 2., 4., 3.],
        [1., 1., 2., 3., 0., 2., 2., 1.],
        [1., 1., 2., 3., 0., 2., 2., 1.],
        [1., 1., 2., 3., 0., 2., 2., 1.]]).cuda()
x = torch.index_select(x, 0, edge_index[0]).float().cuda()
index = edge_index[1].cuda()
origin_index = index.unsqueeze(-1)
origin_index = origin_index.expand_as(x)
new_out = torch.zeros(size=(int(index.max())+1, x.size(1))).cuda()
print("edge_index")
print(edge_index)
print("x")
print(x)
print("index:")
print(index)
print("new_out")
print(new_out)
print("expand_index")
print(origin_index)
origin_test = new_out.scatter_add(0, origin_index, x)
print("origin_test")
print(origin_test)
c_new_out = torch.ops.my_scatter.scatter_sum(x, index, 0, new_out, int(index.max())+1)
print("c_new_out")
print(c_new_out)
# result = torch.equal(c_new_out, origin_test)
# print("运行结果:", result)
# print("内存地址", c_new_out == origin_test)
pass

