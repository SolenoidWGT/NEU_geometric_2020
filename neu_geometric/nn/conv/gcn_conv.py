import torch
from torch.nn import Parameter
from my_scatter import scatter_add
from neu_geometric.nn.conv import MessagePassing
from neu_geometric.utils import add_remaining_self_loops

from ..inits import glorot, zeros
from neu_geometric.utils import scatter_
from neu_geometric.divide.divide import make_mask

class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, to_divide=False,
                 nodes_block=None, edge_block=None, **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # 分块所需参数
        self.to_divide = to_divide
        self.blocks_node_index = nodes_block
        self.blocks_edge_index = edge_block

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None


    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        # edge_weight即为边的特征矩阵，注意不是权重矩阵
        # 如果edge_weight为None，则系统自动赋值为一个和边数量同为的的全1向量
        # 如果边没有特征，则每一条边的特征为1
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        # 为coo格式的edge_index添加自环边，相应的对edge_weight添加新边的特征
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        # 计算各顶点的出度
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # 根据GCN论文公式
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0


        # ***********************在第一次norm时需要修改分块的edge_index,只执行一次！********
        if self.to_divide:
            print("norm,to_divide!!!!")
            self.blocks_edge_index = []
            for item in self.blocks_node_index:
                edge_mask = make_mask(item, edge_index, 1)
                # 获得各block的入边index
                self.blocks_edge_index.append(edge_index[:, edge_mask])

        # 计算： \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def aggregate(self, inputs, index, dim_size, x_input):  # pragma: no cover
        old_size = x_input.size()
        # 注意index是列方向上的index（即终点列表）
        if self.to_divide is False:
            return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)
        else:
            from torch_scatter import scatter
            block_num = len(self.blocks_node_index)
            for i in range(block_num):
                node_list_i, edge_index_i, x_i = self.blocks_node_index[i], self.blocks_edge_index[i], \
                                                 x_input[self.blocks_node_index[i]]
                index_map_i = dict(zip(node_list_i.numpy().tolist(), list(range(len(node_list_i)))))
                for j in range(block_num):
                    node_list_j, edge_index_j, x_j = self.blocks_node_index[j], self.blocks_edge_index[j], \
                                                     x_input[self.blocks_node_index[j]]
                    index_map_j = dict(zip(node_list_j.numpy().tolist(), list(range(len(node_list_j)))))
                    index_now = []
                    index_now_j = []
                    # 获取参与scatter的j遍历的结点
                    for pos, item in enumerate(edge_index_i[0]):
                        if item.item() in index_map_j:
                            index_now_j.append(index_map_j[item.item()])
                            index_now.append(edge_index_i[1, pos].item())

                    if len(index_now) == 0 or len(index_now_j) == 0:
                        continue
                    index_now_j = torch.tensor(index_now_j).long()
                    x = x_j.index_select(0, index_now_j)

                    # 为index输入从0到index元素种类个数-1个的范围给index重新编号
                    for idx in range(len(index_now)):
                        index_now[idx] = index_map_i[index_now[idx]]
                    # 注意新index_now的数据类型和设备所在
                    index_now = torch.tensor(index_now)
                    scatter(src=x, index=index_now, dim=0, out=x_i, reduce="sum")

                node_list_i = node_list_i.unsqueeze(-1)
                node_list_i = node_list_i.expand(size=(node_list_i.size(0), x_input.size(1)))
                x_input.scatter_(dim=0, index=node_list_i, src=x_i)
            return x_input


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
