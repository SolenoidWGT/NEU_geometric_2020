import torch

"""
基本目标：根据节点划分的结果，将edge_index的边按照划分结果
重新组合
"""
edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [1, 2, 3, 0, 3, 0, 3, 0, 1, 2]])
edge_attr = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]).unsqueeze(-1)
# one_div_src_nodes = torch.tensor([0, 3, 1])
# print("all_edge_attr:\n", edge_attr)
# print("one_div_nodes:\n", one_div_src_nodes.unsqueeze(-1))
# mask = torch.eq(one_div_src_nodes.unsqueeze(-1), edge_index[0])
# print("mask\n:", mask)
# one_div_d_node = torch.masked_select(edge_index[1], mask[0])
# print("one_div_d_node:\n", one_div_d_node)
# print("获得新的egd_index:", torch.cat((one_div_src_nodes.unsqueeze(0), one_div_d_node.unsqueeze(0)), 0))
# print("获得划分后小组中边的特征", edge_attr[mask[0]])


def stack_bool_or(mask):
    assert mask.dim() < 3
    if mask.dim() == 1:
        return mask
    new_mask = mask[0].clone()
    for i in range(1, mask.size(0)):
        new_mask += mask[i]  # 对每行进行或操作
    return new_mask


def make_mask(div_node, e_index, direction):
    return stack_bool_or(torch.eq(div_node.unsqueeze(-1), e_index[direction]))


# 划分后的节点组1号和2号
div_node_1 = torch.tensor([0, 2])
div_node_2 = torch.tensor([1, 3])
# 制作结点组的源点mask, 和相应的终点mask
# 注意这里利用了广播性质
# stack_bool将各个结点的mask压缩到1维
mask_1_src = make_mask(div_node_1, edge_index, 0)
mask_2_end = make_mask(div_node_2, edge_index, 1)
# 获得广义邻接矩阵src和end交点的边的列表
select_edge_mask = mask_1_src * mask_2_end  # 与操作
print(select_edge_mask.unsqueeze(0))
select_edge = edge_index[:, select_edge_mask]  # tensor([[0, 0, 2], [1, 3, 3]])
select_edge_attr = edge_attr[select_edge_mask, :]  # tensor([[100],[102], [106]])
# 获得结点组1号和2号的所有出边的终点
# div_node_1_end = torch.masked_select(edge_index[1], mask_1)
# div_node_2_end = torch.masked_select(edge_index[1], mask_2)
pass
