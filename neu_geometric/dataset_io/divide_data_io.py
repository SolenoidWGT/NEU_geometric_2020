import torch
import os
import os.path as osp


def test_save_divide(node_blocks, edge_block, x: torch.tensor, edge_attr: torch.tensor = None):
    nodes_dict = dict(zip(list(range(len(node_blocks))), node_blocks))
    edge_block = dict(zip(list(range(len(edge_block))), edge_block))
    for key, value in nodes_dict.items():
        torch.save({"node": nodes_dict[key],
                    "edge": edge_block[key],
                    "x": x[nodes_dict[key]],
                    },
                   '/home/wgt/PycharmProjects/final_project/raw_data/test_divide_data/'+'data_'+str(key)+'.pth')


def load_divide(path, file=None):
    if file is None:
        for root, _, files in os.walk(path):
            for i, f in enumerate(files):
                data = torch.load(osp.join(path, 'data_'+str(i)+'.pth'), map_location='cpu')
    else:
        data = torch.load(osp.join(path, file), map_location='cpu')
        x = data['x']
        x_index = data['node']
        x = x[x_index]
        x_index[1]


def make_dirt(x_index):
    index_map = dict(zip(list(range(len(x_index))), x_index))


from torch_scatter import scatter
def fake_for(block_num, data):
    from torch_scatter import scatter
    block_num = len(blocks_node_index)
    for i in range(block_num):
        node_list_i, edge_index_i, x_i = blocks_node_index[i], block_edge_index[i], data.x[blocks_node_index[i]]
        for j in range(block_num):
            node_list_j, edge_index_j, x_j = blocks_node_index[j], block_edge_index[j], data.x[blocks_node_index[j]]
            index_map_j = dict(zip(node_list_j.numpy().tolist(), list(range(len(node_list_j)))))
            x = None
            index_now = []
            # 获取参与scatter的j遍历的结点
            for item in edge_index_i[0]:
                if item.item() in index_map_j:
                    if x is None:
                        x = x_j[index_map_j[item.item()]].unsqueeze(0)
                    else:
                        x = torch.cat((x, x_j[index_map_j[item.item()]].unsqueeze(0)), dim=0)
                    index_now.append(item.item())
            # 为index输入从0到index元素种类个数-1个的范围给index重新编号
            no_repeat_index = list(set(index_now))
            no_repeat_index.sort()
            no_repeat_index_map = dict(zip(no_repeat_index, list(range(len(no_repeat_index)))))
            for idx in range(len(index_now)):
                index_now[idx] = no_repeat_index_map[index_now[idx]]
            # 注意新index_now的数据类型和设备所在
            index_now = torch.tensor(index_now)
            scatter(src=x, index=index_now, dim=0, out=x_i, reduce="sum")





