import networkx as nx
import matplotlib.pyplot as plt
from neu_geometric.utils.isolated import contains_isolated_nodes


if __name__ == "__main__":

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
    edge_index = data.edge_index.permute(1, 0).numpy().tolist()
    H = nx.path_graph(2708)
    H.add_edges_from(edge_index)
    # pos = nx.circular_layout(H)  # 环形布局
    nx.draw(H, node_size=1,width=0.2, with_labels=False)
    plt.show()



    pass
