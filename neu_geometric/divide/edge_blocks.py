import torch


def stack_bool_or(mask):
    assert mask.dim() < 3
    if mask.dim() == 1:
        return mask
    new_mask = mask[0].clone()
    for i in range(1, mask.size(0)):
        new_mask += mask[i]  # 对每行进行或操作
    return new_mask


def make_mask(div_node: torch.Tensor, e_index, direction):
    return stack_bool_or(torch.eq(div_node.unsqueeze(-1), e_index[direction]))


def get_edge_blocks(div_node_src, div_node_end, edge_index, edge_attr):
    mask_1_src = make_mask(div_node_src, edge_index, 0)
    mask_2_end = make_mask(div_node_end, edge_index, 1)
    select_edge_mask = mask_1_src * mask_2_end  # 与操作
    # 获得edge_blocks
    select_edge = edge_index[:, select_edge_mask]  # tensor([[0, 0, 2], [1, 3, 3]])
    # 获得edge_blocks对应边的特征矩阵
    select_edge_attr = edge_attr[select_edge_mask, :]  # tensor([[100],[102], [106]]
    return select_edge, select_edge_attr
