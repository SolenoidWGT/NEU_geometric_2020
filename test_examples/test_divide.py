import torch


# def stack_bool_or(mask):
#     assert mask.dim() < 3
#     if mask.dim() == 1:
#         return mask
#     new_mask = mask[0].clone()
#     for i in range(1, mask.size(0)):
#         new_mask += mask[i]  # 对每行进行或操作
#     return new_mask
#
#
# def make_mask(div_node: torch.Tensor, e_index, direction):
#     return stack_bool_or(torch.eq(div_node.unsqueeze(-1), e_index[direction]))
#
#
# def change_sign(index: torch.tensor):
#     pass
#
#
# def neighbor_sampling_fast(edge_index: torch.Tensor):
#     temp_index = edge_index.clone()
#     # 对于所有的edge_index加1, 避免index为0干扰后面的计算
#     temp_index += 1
#     # 初始化
#     mask = make_mask(temp_index[0, 4], temp_index, 0)
#     node_num = 0
#     p = 2
#     block_ = []
#     while temp_index[0].max() > 0:
#         # 统计扩散新增加的结点数量
#         node_num += len(set(temp_index[0, mask][temp_index[0, mask] > 0.].tolist()))
#         # 置反
#         whe = temp_index[0][mask]
#         temp_index[0][mask] = whe.where(whe < 0, -whe)
#         if node_num >= p:
#             block_.append(mask)
#             ttt = temp_index[0, temp_index[0, :] > 0][0]
#             mask = make_mask(temp_index[0, temp_index[0, :] > 0][0], temp_index, 0)
#             node_num = 0
#             continue
#         # 扩散
#         mask += make_mask(temp_index[1][mask], temp_index, 0)
#     block_.append(mask)
#     return block_
#
#
# # 作用：将mask中为True的index值置反
# def reverse(e_index: torch.Tensor, mask: torch.Tensor):
#     whe = e_index[0][mask]
#     e_index[0][mask] = whe.where(whe < 0, -whe)
#
#
# # 统计在为一个结点制作mask后，新增可供扩展的的结点数量
# def count_(e_index: torch.Tensor, mask: torch.Tensor):
#     # 统计扩散新增加的结点数量
#     return len(set(e_index[0, mask][e_index[0, mask] > 0.].tolist()))
#
#
# # 作用：尝试遍历1个结点(相当于遍历该结点)
# def try_visit(div_node: torch.Tensor, e_index: torch.tensor, mask: torch.Tensor = None):
#     if div_node is None:
#         return False, None
#     # 掩模制作
#     new_mask = torch.eq(div_node.unsqueeze(-1), e_index[0])
#     # 有效扩展确认, 如果一个new_mask的count_new为0，则这个结点就已经被遍历过了
#     if count_(e_index, new_mask) == 0:
#         return False, mask
#     # 只有被置反才说明真正被遍历到
#     reverse(e_index, new_mask)
#     # 累加
#     return True, new_mask
#
#
# # 在edge_index寻找还未遍历的结点
# def find_new(e_index: torch.Tensor):
#     positive_index = e_index[0, e_index[0, :] > 0]
#     if positive_index.size(0) == 0:
#         return None
#     else:
#         return positive_index[0]
#
# # 初始化新block_mask
# def init_block():
#     pass
#
#
# # 不能存在自环！！
# # 必须是无向图！！
# def neighbor_sampling(edge_index: torch.Tensor):
#     temp_index = edge_index.clone()
#     # 对于所有的edge_index加1, 避免index为0干扰后面的计算
#     temp_index += 1
#     # 初始化
#     _, mask = try_visit(temp_index[0, 0], temp_index)
#     node_num = 1
#     p = 2  # p不能为1!!
#     block_ = []
#     stride = 2
#     end_flag = False
#     while temp_index[0].max() > 0:
#         # 当前mask的等待广播的列表不为空
#         # def make_mask(div_node: torch.Tensor, e_index, direction):
#         while make_mask(temp_index[1][mask], temp_index, 0).max() > 0:
#             if node_num == p:
#                 block_.append(mask)
#                 node_num = 0
#                 end_flag = True
#                 break
#             wait_node = temp_index[1][mask]  # 待广播的列表
#             for node in wait_node:
#                 re, t_mask = try_visit(node, temp_index, mask)
#                 if re is False:
#                     continue
#                 else:
#                     node_num += 1
#                     mask += t_mask
#                     if node_num == p:
#                         block_.append(mask)
#                         end_flag = True
#                         node_num = 0
#                         break
#             if end_flag is True:
#                 break
#         # 建立新的block的mask
#         if end_flag is True:
#             _, mask = try_visit(find_new(temp_index), temp_index)
#             if mask is None:
#                 print("所有结点已经分配完毕，结束")
#                 break
#             end_flag = False
#             node_num = 1
#         else:
#             _, t_mask = try_visit(find_new(temp_index), temp_index)
#             if t_mask is None:
#                 print("所有结点已经分配完毕，结束")
#                 break
#             mask += t_mask
#             node_num += 1
#     block_.append(mask)
#     return block_



# [1, 2, 3, 0, 3, 0, 3, 0, 1, 2
# edge_index_ = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3], [0, 1, 2, 3, 1, 0, 3, 2, 0, 3, 3, 0, 1, 2]])
# edge_attr = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]).unsqueeze(-1)

edge_index_ = torch.tensor([[1, 2, 2, 2, 2, 4, 4, 5, 6, 6], [0, 1, 3, 4, 6, 2, 5, 2, 2, 7]])
edge_index_add_self_loop = torch.tensor([[0,1,1,2,2,2,2,2,3,4,4,4,5,5,6,6,6,7], [0,0,1,1,2,3,4,6,3,2,4,5,2,5,2,6,7,7]])
edge_index_add_self_loop_attr = torch.ones(size=(edge_index_add_self_loop.size(1), 1))
edge_index_no_dirt = torch.tensor([[0,1,1,2,2,2,2,2,3,4,4,5,5,6,6,7],[1,0,2,1,3,4,5,6,2,2,5,2,4,2,7,6]])
x_small = torch.randint(low=0, high=5, size=(8, 5)).float()
"""
 tensor([[3., 4., 0., 4., 4.],
        [4., 3., 2., 0., 1.],
        [2., 1., 2., 2., 2.],
        [4., 2., 1., 1., 0.],
        [4., 0., 0., 3., 0.],
        [3., 0., 3., 2., 4.],
        [4., 3., 4., 0., 1.],
        [3., 3., 1., 2., 3.]])
"""

# block = neighbor_sampling(edge_index_no_dirt)
from neu_geometric.utils.loop import remove_self_loops
from neu_geometric.divide.divide import divide_blocks, mini_divide
from neu_geometric.dataset_io.divide_data_io import load_divide,test_save_divide
if __name__ == "__main__":
    # p = 4
    # blocks_node_index, blocks_edge_index = divide_blocks(edge_index_add_self_loop, p)
    # btt, ett = mini_divide(edge_index_no_dirt, p)




    from neu_geometric.utils.isolated import contains_isolated_nodes
    import os.path as osp
    import argparse

    import torch
    from neu_geometric.dataset_script import Planetoid
    import neu_geometric.transforms as T
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')
    args = parser.parse_args()
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'raw_data', dataset)
    pass
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]
    node_number = 2708
    p= 200
    re = contains_isolated_nodes(data.edge_index, 2708)
    # blocks_node_index, block_edge_index = divide_blocks(data.edge_index.cpu(), p, 2708)
    # test_save_divide(blocks_node_index, block_edge_index, data.x)


    blocks_node_index, block_edge_index = divide_blocks(edge_index_, p=2, node_num=8)
    # print("edge_index_\n", edge_index_)
    # print("blocks_node_index\n", blocks_node_index)
    # print("block_edge_index\n", block_edge_index)
    # print("x_small\n", x_small)

    from torch_scatter import scatter
    block_num = len(blocks_node_index)
    for i in range(block_num):
        node_list_i, edge_index_i, x_i = blocks_node_index[i], block_edge_index[i], x_small[blocks_node_index[i]]
        index_map_i = dict(zip(node_list_i.numpy().tolist(), list(range(len(node_list_i)))))
        for j in range(block_num):
            node_list_j, edge_index_j, x_j = blocks_node_index[j], block_edge_index[j], x_small[blocks_node_index[j]]
            index_map_j = dict(zip(node_list_j.numpy().tolist(), list(range(len(node_list_j)))))
            x = None
            index_now = []
            # 获取参与scatter的j遍历的结点
            for pos, item in enumerate(edge_index_i[0]):
                if item.item() in index_map_j:
                    if x is None:
                        x = x_j[index_map_j[item.item()]].unsqueeze(0)
                    else:
                        x = torch.cat((x, x_j[index_map_j[item.item()]].unsqueeze(0)), dim=0)
                    index_now.append(edge_index_i[1, pos].item())
            # 为index输入从0到index元素种类个数-1个的范围给index重新编号
            for idx in range(len(index_now)):
                index_now[idx] = index_map_i[index_now[idx]]
            # 注意新index_now的数据类型和设备所在
            index_now = torch.tensor(index_now)
            if index_now.shape[0] != 0:
                scatter(src=x, index=index_now, dim=0, out=x_i, reduce="sum")

    # load_divide("/home/wgt/PycharmProjects/final_project/raw_data/test_divide_data/")
    pass