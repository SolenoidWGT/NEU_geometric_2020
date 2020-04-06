from .edge_blocks import make_mask
from ..utils.loop import select_self_loops
from ..utils.undirected import to_undirected
from torch_sparse import coalesce
from ..utils.num_nodes import maybe_num_nodes
from ..utils.undirected import is_undirected
import torch

"""
我们假设图数据的edge_index可以以coo形式的稀疏矩阵加载到内存中
"""


# 作用：将mask中为True的index值置反
def reverse(e_index: torch.Tensor, mask: torch.Tensor):
    whe = e_index[0][mask]
    e_index[0][mask] = whe.where(whe < 0, -whe)


# 统计在为一个结点制作mask后，新增可供扩展的的结点数量
def count_(e_index: torch.Tensor, mask: torch.Tensor):
    # 统计扩散新增加的结点数量
    # return len(set(e_index[0, mask][e_index[0, mask] > 0.].cpu().numpy().tolist()))
    return deduplication_nodes(e_index, mask).size(0)


# 作用：尝试遍历1个结点(相当于遍历该结点)
def try_visit(div_node: torch.Tensor, e_index: torch.tensor, mask: torch.Tensor = None):
    if div_node is None:
        return False, None
    # 掩模制作
    new_mask = torch.eq(div_node.unsqueeze(-1), e_index[0])
    # # 有效扩展确认, 如果一个new_mask的count_new为0，则这个结点就已经被遍历过了
    # if count_(e_index, new_mask) == 0:
    #     return False, mask
    if new_mask.sum() == 0:
        return False, mask
    # 只有被置反才说明真正被遍历到
    reverse(e_index, new_mask)
    return True, new_mask


# 在edge_index寻找还未遍历的结点
def find_new(e_index: torch.Tensor):
    positive_index = e_index[0, e_index[0, :] > 0]
    if positive_index.size(0) == 0:
        return None
    else:
        return positive_index[0]


# 不能存在自环！！
# 必须是无向图！！
# p不能为1!!
def neighbor_sampling(edge_index: torch.Tensor, p: int, append_last_flag: bool):
    t_index = edge_index.clone()
    # 对于所有的edge_index加1, 避免index为0干扰后面的计算
    t_index += 1
    # 初始化
    _, mask = try_visit(t_index[0, 0], t_index)
    node_num = 1
    block_ = []
    end_flag = False
    # while t_index[0].max() > 0:
    while True:
        # 当前mask的等待广播的列表不为空
        if node_num >= p:
            block_.append(mask)
            node_num = 0
            end_flag = True
        if end_flag is False:
            while make_mask(t_index[1][mask], t_index, 0).max() > 0:
                wait_node = t_index[1][mask]  # 待广播的列表
                for node in wait_node:
                    re, t_mask = try_visit(node, t_index, mask)
                    if re is False:
                        continue
                    else:
                        node_num += 1
                        mask += t_mask
                        if node_num >= p:
                            block_.append(mask)
                            end_flag = True
                            node_num = 0
                            break
                if end_flag is True:
                    break
        next_node = find_new(t_index)
        if next_node is None:
            # print("所有结点已经分配完毕，结束")
            break
        # 建立新的block的mask
        if end_flag is True:
            _, mask = try_visit(next_node, t_index)
            end_flag = False
            node_num = 1
        else:
            _, t_mask = try_visit(next_node, t_index)
            mask += t_mask
            node_num += 1
    if append_last_flag:
        block_.append(mask)
    print("neighbor_sampling ended!")
    return block_


# 快速邻居采样，无法保证每一组的结点数量等于P，有可能会比P大很多
# 但是这个方法的时间复杂度非常低
def neighbor_sampling_fast(edge_index: torch.Tensor):
    t_index = edge_index.clone()
    # 对于所有的edge_index加1, 避免index为0干扰后面的计算
    t_index += 1
    # 初始化
    mask = make_mask(t_index[0, 4], t_index, 0)
    node_num = 0
    p = 2
    block_ = []
    while t_index[0].max() > 0:
        # 统计扩散新增加的结点数量
        # node_num += len(set(t_index[0, mask][t_index[0, mask] > 0.].cpu().numpy().tolist()))
        node_num += deduplication_nodes(t_index, mask).size(0)
        # 置反
        whe = t_index[0][mask]
        t_index[0][mask] = whe.where(whe < 0, -whe)
        if node_num >= p:
            block_.append(mask)
            # ttt = temp_index[0, temp_index[0, :] > 0][0]
            mask = make_mask(t_index[0, t_index[0, :] > 0][0], t_index, 0)
            node_num = 0
            continue
        # 扩散
        mask += make_mask(t_index[1][mask], t_index, 0)
    block_.append(mask)
    return block_


def show_():
    pass


# coalesce去重
def deduplication_nodes(edge_index, mask):
    """
    coalesce这个函数本来是用来删除图中重复的边的，但是如果将等待去重的
    列表看作是row，再设置一个全0的row，进行coalesce操作后返回的就是一个
    消除重复元素的tensor，而coalesce是一个GPU操作，避免了数据从divice到host的
    拷贝，大大提高了效率
    :param edge_index:
    :param mask:
    :return:
    """
    dup_index = edge_index[0, mask].clone()
    zero_mask = torch.zeros_like(dup_index)
    fit_coalesce_index = torch.stack((dup_index, zero_mask), dim=0)
    dedup_index, _ = coalesce(fit_coalesce_index, None, m=1, n=1)
    return dedup_index[0]




def divide_blocks(edge_index_: torch.Tensor, p: int, node_num=-1):
    """
    divide_block得到的block主要是为了将庞大的edge_attr矩阵和x矩阵划分到
    一些可以放入内存大小的矩阵块
    """
    edge_index = edge_index_.clone()
    # 估计结点总数
    if node_num == -1:
        node_num = maybe_num_nodes(edge_index)
    # 去掉自环，同时保留自环的特征，在最后处理结束将自环加回去
    edge_index_t = select_self_loops(edge_index)
    # 转换为无向图
    if not is_undirected(edge_index):
        edge_index_t = to_undirected(edge_index_t)
    # 进行分块划分
    blocks_edge_mask = neighbor_sampling(edge_index_t, p, False if node_num % p == 0 else True)
    blocks_node_index = []
    blocks_edge_index = []
    for item in blocks_edge_mask:
        # 此处使用coalesce更快还是用set处理更快？肯定是使用coalesce快,
        # 使用set的话：sorted(list(set(edge_index[0, item].cpu().numpy().tolist()))))
        # *gpu上的tensor不能直接转为numpy，加b=a.cpu().numpy(), 必须要从数据从GPU拷贝回CPU
        # 这是绝对不能被容忍的

        # 可以使用这种方法索引特征矩阵x:  aa[torch.tensor([0,3,4]),:]
        """
        现在有一个问题，特征矩阵x如果无法放入内存，那我获取了特征矩阵的分块索引该怎么用？？？
        我不管，我给你划分好了索引，剩下分配出一个可以放入内存的特征x矩阵是用户的事情
        """
        # 去重, 获得x特征矩阵的index
        x_block_index = deduplication_nodes(edge_index_t, item)
        blocks_node_index.append(x_block_index)
        # 获取原edge_index的mask
        edge_mask = make_mask(x_block_index, edge_index, 1)
        # 获得各block的入边index
        blocks_edge_index.append(edge_index[:, edge_mask])
    return blocks_node_index, blocks_edge_index


def mini_divide(edge_index: torch.Tensor(), p: int):
    """
    divide_block得到的block主要是为了将庞大的edge_attr矩阵和x矩阵划分到
    一些可以放入内存大小的矩阵块
    """
    # 估计结点总数
    node_nums = maybe_num_nodes(edge_index)
    # 进行分块划分
    blocks_edge_mask = neighbor_sampling(edge_index, p, False if node_nums % p == 0 else True)
    blocks_node_index = []
    blocks_edge_index = []
    for item in blocks_edge_mask:
        x_block_index = deduplication_nodes(edge_index, item)
        blocks_node_index.append(x_block_index)
        # 获取原edge_index的mask
        edge_mask = make_mask(x_block_index, edge_index, 0)
        blocks_edge_index.append(edge_index[:, edge_mask])
    return blocks_node_index, blocks_edge_index






