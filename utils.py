import pandas as pd
import networkx as nx
from sklearn.metrics import adjusted_rand_score
import time
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csgraph
import copy
import matplotlib.pyplot as plt
from collections import deque
from tqdm.rich import tqdm


class Node(object):
    def __init__(self, id, score):
        self.id = id
        self.score = score

    def __lt__(self, other):
        return self.score > other.score


def rbf_kernel(X, Y=None, gamma=None):

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K


def neiborhood_insertion(N: list[list], data, labels):
    count = 0
    for i, x_k in enumerate(N):
        count += 1
        if labels[data] == labels[x_k[0]]:
            x_k.append(data)
            return i, count
    N.append([data])
    id = len(N)-1 if len(N) > 0 else 0
    return id, count


def refine_dislike(ET: nx.Graph, data, real_labels, R: list, e, att):
    e = e[0]
    ET.remove_edge(*e)
    n, _ = e
    Find = False
    rl = [(r, np.linalg.norm(data[n]-data[r])) for r in R]
    rl = [r for r, _ in sorted(rl, key=lambda x: x[1])]
    count: int = 0
    for r in rl:
        count += 1
        if real_labels[n] == real_labels[r]:
            Find = True
            ET.add_edge(n, r)
            # if att[n] > att[r]:
            #     ET.add_edge(n, r)
            # else:
            #     ET.add_edge(r, n)
            #     R.remove(r)
            #     R.append(n)
            break
    if not Find:
        R.append(n)
    return count


def get_rank_by_layer_and_edge_rank(ET: nx.Graph, roots, data: np.array):
    # bfs ET
    # ET = ET.to_undirected()
    assert len(ET.nodes) == len(data), 'len of ET nodes and data is not equal'
    # ET = nx.reverse(ET)
    R = []
    bfs_layers = dict(enumerate(nx.bfs_layers(ET, roots)))
    flayer = bfs_layers.pop(0)

    print(f'flayer is {flayer}')

    for layers, nodes in bfs_layers.items():
        # print(f'layer {layers} has {len(nodes)} nodes')
        # R.extend(nodes)
        print(f'layer {layers} has {len(nodes)} nodes')
        # nn = []
        # R.extend(nodes)
        # for n in nodes:
        #     dis = 0
        #     k = 0
        #     edges = list(ET.edges(n))
        #     if edges:
        #         for ei, ej in edges:
        #             dis += np.linalg.norm(data[ei]-data[ej])
        #             k += 1
        #         nn.append([n, dis/k])
        #     else:
        #         nn.append([n, 0])
        #     # nn.append([n, dis/k])
        # R.extend([n for n, d in sorted(nn, key=lambda x: x[1], reverse=True)])
    # R[0], R[-1] = R[-1], R[0]
    assert len(R) == len(ET.nodes) - \
        len(roots), f'{len(R) =} is not equal to {
            len(ET.nodes) - len(roots) =}, {len(roots) =} of ET nodes'
    print(f'{len(R)=}')
    # R.reverse()
    # R.remove(roots[0])
    return R


def get_node_attrs_by_distance_from_root(G: nx.Graph, data: np.array, roots: list):
    # attrs: dict[int, dict] = {}
    attrs = np.zeros(len(G.nodes))
    # attrs[roots[0]] = {'distance': 0}
    # edges = set(G.edges(roots[0]))
    successor = []
    # bfs ET
    queue = deque(roots)
    alpha = 1
    max_dis = 0
    while queue:
        node = queue.popleft()
        successor = list(G.neighbors(node))
        queue.extend(successor)
        for n in successor:
            # if n in attrs:
            #     continue
            # attrs[n] = {'distance': np.linalg.norm(
            #     data[n]-data[node]) + attrs[node]['distance']}
            dist = np.linalg.norm(data[n]-data[node])
            attrs[n] = alpha*dist + attrs[node]
    # nx.set_node_attributes(G, attrs)
    # max_dis = np.max(attrs)
    # attrs = attrs/max_dis
    # for n in range(len(attrs)):
    #     attrs[n] += G.out_degree(n)
    return attrs


def update_link_counts(linkCounts: dict[str, list], flage: bool):
    ml = linkCounts['ML'][-1] + int(flage)
    cl = linkCounts['CL'][-1] + int(not flage)
    linkCounts['ML'].append(ml)
    linkCounts['CL'].append(cl)


def merge_roots(ET: nx.Graph, roots: list):
    if len(roots) == 1:
        return roots

    if ET.in_degree(roots[0]) > ET.in_degree(roots[1]):
        ET.add_edge(roots[1], roots[0])
        roots.pop(1)
        # roots = [roots[0]]
    else:
        ET.add_edge(roots[0], roots[1])
        roots.pop(0)
        # roots = [roots[1]]
    return merge_roots(ET, roots)


def get_predict_labels(Graph: nx.Graph):
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    predict_labels = np.zeros(len(Graph.nodes), dtype=int)

    for i, s in enumerate(S[1:], 1):
        predict_labels[list(s.nodes)] = i
    return predict_labels


def plot_dataframe(savePath: str = 'result/pic/linkCounts'):
    """
    画出DataFrame的折线图，以iter为X轴，ML和CL为Y轴。

    参数:
    df (pd.DataFrame): 包含iter, ML, CL列的DataFrame
    """
    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({'figure.max_open_warning': 0})
    # plt.figure(figsize=(10, 6))
    os.makedirs(savePath, exist_ok=True)
    for file in os.listdir('result/other'):
        df = pd.read_csv(f'result/other/{file}')
        # 绘制ML列的折线图
        plt.plot(df['iter'], df['ML'], label='ML')

        # 绘制CL列的折线图
        plt.plot(df['iter'], df['CL'], label='CL')

        # 添加标题和标签
        plt.title('Count of ML and CL')
        # plt.xlabel('Query', fountsize=20)
        # plt.ylabel('Count', fountsize=20)
        plt.xlabel('Query')
        plt.ylabel('Count')

        # 添加图例
        plt.legend()
        # 显示网格
        plt.grid(True)

        # 显示图表
        # plt.show()
        plt.savefig(f'{savePath}/{file.split(".")[0]}.png')
        plt.cla()


def draw_graph_e(ARIpath: str, pathToSavePic: str = 'result/pic', file='', remove=False, matric='ARI'):
    print('save result to pic')
    # import scienceplots
    # plt.style.use(['science'])
    # algpath = ['OUR-kmeans-100', 'DSL', 'ADP', 'ADPE', 'COBRAS']
    import scienceplots
    plt.style.use(['science', 'ieee'])

    algpath = ['Ablation/OUR-9-t100', 'ADP',
               'ADPE', 'COBRAS', 'COBRA', 'FFQS', 'MinMax']
    color = {'Ablation/OUR-9-t100': 'r', 'ADP': 'g',
             'ADPE': 'b', 'COBRAS': 'y', 'COBRA': 'm',
             'FFQS': 'c', 'MinMax': 'k'}
    # algpath = os.listdir('result/Ablation')
    # algpath = [f'Ablation/{i}' for i in algpath]
    # algpath = [f'OUR-{i*100}' for i in range(0, 6)]
    # algpath = [f'OUR-kmeans-{i*100}' for i in range(0, 6)]
    # algpath = ['OUR-0'] + ['OUR-100']
    # path = [f'{ARIpath}/{i}' for i in algpath]
    os.makedirs(pathToSavePic, exist_ok=True)

    # plt.style.use(['science', 'ieee'])
    # with plt.style.context(['science', 'ieee']):
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    _maxX = 0
    ARIs = []
    _alg = []
    for i in algpath:
        fpath = f'{ARIpath}/{i}/{file}'
        print(f'load {fpath}')
        if not os.path.exists(fpath):
            continue
        print(f'load {fpath}')
        _alg.append(i)
        ARI = pd.read_csv(fpath)
        ARIs.append(ARI)
        # 设置线条的颜色
        # 平滑曲线
        _maxX = max(_maxX, ARI['interaction'].max())

    for i, ARI in enumerate(ARIs):
        # 将ARI['interaction']，ARI是datafrem 增加一个数据
        X = ARI['interaction'].tolist()

        # X.append(_maxX)
        # print(X)
        window_size = 20
        # y = ARI['ari'].rolling(window=window_size, center=True).mean()
        y = ARI['ari'].tolist()
        # y = y.tolist()

        # y = np.array(y)
        # # 对进行拟合
        # f = interp1d(X, y, kind='cubic')
        # X = np.linspace(0, _maxX)
        # y = f(X)

        # X = X.tolist()
        X.append(_maxX)
        # y = y.tolist()
        y.append(y[-1])

        ax.plot(X, y,
                label=_alg[i], color=color[_alg[i]])

    # ax.set_xlabel('Quries scale log')
    ax.set_xlabel('Quries')
    ax.set_ylabel(matric)
    ax.set_title(f'{file.split(".")[0]}')
    xlimL = ['banknote.csv', 'breast.csv',
             'Facebook Live Sellers in Thailand.csv', 'iris.csv', 'Statlog.csv', 'thyroid.csv']
    xlim = {'banknote.csv': 400, 'breast.csv': 500,
            'Facebook Live Sellers in Thailand.csv': 6200,
            'iris.csv': 110, 'Statlog.csv': 8000, 'thyroid.csv': 300}

    if file in xlimL:
        ax.set_xlim(0, xlim[file])
    else:
        ax.set_xlim(0, _maxX)
    # ax.set_xscale('log')
    ax.legend()

    path = f'{pathToSavePic}/{file.split(".")[0]}.png'
    # print(f'save  to {path}')
    fig.savefig(path, dpi=600)


def refine_by_h_random(ET: nx.Graph, roots: list, data, real_labels, subroots: list, dataName: str = 'nan', alpha=.5, **kwargs):
    '''
    1. random 
    '''
    print(f'start refine_by_h random')
    linkCounts = {'ML': [0], 'CL': [0]}
    s = time.perf_counter()
    # att = get_node_attrs_by_distance_from_root(ET, data, roots)
    R: list = roots.copy()
    UET: nx.Graph = ET.to_undirected()

    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))

    df: dict[int, list] = {"iter": [0],
                           "interaction": [0], "ari": [ARI], "time": [0]}

    c, i = 0, 0
    edges = list(ET.edges())
    for edge in edges:
        # print(f'{edge=}, {count=},{_max=}')
        # print(f'26 {edge=}')
        a, b = edge
        count = 1

        update_link_counts(linkCounts, real_labels[a] == real_labels[b])

        if real_labels[a] != real_labels[b]:
            count += refine_dislike(ET, data,
                                    real_labels, R, [edge], None)

    #     # ARI = df['ari'][-1]
    #     # if True or i % 2 == 0:
        predict_labels = get_predict_labels(ET)

        ARI = adjusted_rand_score(real_labels, predict_labels)
    #     # ARI = rand_score(real_labels, predict_labels)
        t = time.perf_counter()
    #     # RI = rand_score(real_labels, predict_labels)

        c += count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
    #     # df['ari'].append(RI)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            break

    linkCountsP = 'result/other/'
    os.makedirs(linkCountsP, exist_ok=True)
    lc = pd.DataFrame(linkCounts)
    # rename index name to iter
    lc.index.name = 'iter'
    lc.to_csv(f'{linkCountsP}{dataName}_linkCounts.csv', index=True)

    dfForalpha = kwargs['dfForalpha']
    dfForalpha['dataName'].append(dataName)
    dfForalpha['alpha'].append(alpha)
    dfForalpha['beta'].append(1-alpha)
    dfForalpha['query'].append(c)

    print(f'finash')
    return pd.DataFrame(df), R


def refine_by_h_edge_weight_rank(ET: nx.Graph, roots: list, data, real_labels, subroots: list, dataName: str = 'nan', **kwargs):
    '''
    1. edge_weight_rank the farthest edge first
    '''
    print(f'start refine_by_h edge_weight_rank')
    linkCounts = {'ML': [0], 'CL': [0]}
    s = time.perf_counter()
    # att = get_node_attrs_by_distance_from_root(ET, data, roots)
    R: list = roots.copy()
    UET: nx.Graph = ET.to_undirected()

    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))

    df: dict[int, list] = {"iter": [0],
                           "interaction": [0], "ari": [ARI], "time": [0]}

    c, i = 0, 0
    edges = {}

    for e in ET.edges():
        u, v = e
        edges[e] = np.linalg.norm(data[u]-data[v])

    edge_rank = sorted(edges.items(), key=lambda x: x[1], reverse=True)

    for edge, _ in edge_rank:
        # print(f'{edge=}, {count=},{_max=}')
        # print(f'26 {edge=}')
        a, b = edge
        count = 1

        update_link_counts(linkCounts, real_labels[a] == real_labels[b])

        if real_labels[a] != real_labels[b]:
            count += refine_dislike(ET, data,
                                    real_labels, R, [edge], None)

    #     # ARI = df['ari'][-1]
    #     # if True or i % 2 == 0:
        predict_labels = get_predict_labels(ET)

        ARI = adjusted_rand_score(real_labels, predict_labels)
    #     # ARI = rand_score(real_labels, predict_labels)
        t = time.perf_counter()
    #     # RI = rand_score(real_labels, predict_labels)

        c += count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
    #     # df['ari'].append(RI)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            break

    linkCountsP = 'result/other/'
    os.makedirs(linkCountsP, exist_ok=True)
    lc = pd.DataFrame(linkCounts)
    # rename index name to iter
    lc.index.name = 'iter'
    lc.to_csv(f'{linkCountsP}{dataName}_linkCounts.csv', index=True)

    dfForalpha = kwargs['dfForalpha']
    # dfForalpha['dataName'].append(dataName)
    # dfForalpha['alpha'].append(alpha)
    # dfForalpha['beta'].append(1-alpha)
    # dfForalpha['query'].append(c)

    print(f'finash')
    return pd.DataFrame(df), R


def refine_by_h_edge_weight_rank_chain(ET: nx.Graph, roots: list, data, real_labels, subroots: list, dataName: str = 'nan', **kwargs):
    '''
    1. edge_weight_rank the farthest edge first
    '''
    print(f'start refine_by_h edge_weight_rank')
    linkCounts = {'ML': [0], 'CL': [0]}
    s = time.perf_counter()
    # att = get_node_attrs_by_distance_from_root(ET, data, roots)
    R: list = roots.copy()
    UET: nx.Graph = ET.to_undirected()
    # RET = nx.reverse(ET)

    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))

    df = {"iter": [0],
          "interaction": [0], "ari": [ARI], "time": [0]}

    c, i = 0, 0
    attr = np.zeros(len(ET.nodes))
    eweight = {}

    bfs_layers = dict(enumerate(nx.bfs_layers(UET, roots)))
    bfs_layers.pop(0)
    _maxdist = 0
    _mixdegree = 0
    for layers, nodes in bfs_layers.items():
        for n in nodes:
            parent = list(ET.neighbors(n))[0]
            # parant = list(RET.neighbors(n))
            # if not e:
            #     continue
            e = (n, parent)
            d = np.linalg.norm(data[e[0]]-data[e[1]])

            _mixdegree = max(_mixdegree, ET.in_degree(
                e[0]) + ET.in_degree(e[1]))
            # attr[n] = attr[e[1]] + d
            attr[n] = d/layers
            _maxdist = max(_maxdist, attr[n])
            eweight[e] = attr[n]

    edges = {e: eweight[e]/_maxdist +
             ET.in_degree(e[0]) + ET.in_degree(e[1])/_mixdegree for e in eweight.keys()}

    edge_rank = sorted(edges.items(), key=lambda x: x[1], reverse=True)

    for edge, _ in edge_rank:
        # print(f'{edge=}, {count=},{_max=}')
        # print(f'26 {edge=}')
        a, b = edge
        count = 1

        update_link_counts(linkCounts, real_labels[a] == real_labels[b])

        if real_labels[a] != real_labels[b]:
            count += refine_dislike(ET, data,
                                    real_labels, R, [edge], None)

    #     # ARI = df['ari'][-1]
    #     # if True or i % 2 == 0:
        predict_labels = get_predict_labels(ET)

        ARI = adjusted_rand_score(real_labels, predict_labels)
    #     # ARI = rand_score(real_labels, predict_labels)
        t = time.perf_counter()
    #     # RI = rand_score(real_labels, predict_labels)

        c += count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
    #     # df['ari'].append(RI)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            break

    linkCountsP = 'result/other/'
    os.makedirs(linkCountsP, exist_ok=True)
    lc = pd.DataFrame(linkCounts)
    # rename index name to iter
    lc.index.name = 'iter'
    lc.to_csv(f'{linkCountsP}{dataName}_linkCounts.csv', index=True)

    dfForalpha = kwargs['dfForalpha']
    # dfForalpha['dataName'].append(dataName)
    # dfForalpha['alpha'].append(alpha)
    # dfForalpha['beta'].append(1-alpha)
    # dfForalpha['query'].append(c)

    print(f'finash')
    return pd.DataFrame(df), R


def refine_by_h_edge_mlit_nei(ET: nx.Graph, roots: list, data, real_labels, subroots: list, dataName: str = 'nan', **kwargs):
    '''
    1. edge_weight_rank the farthest edge first
    2. 
    '''
    print(f'{dataName} start refine_by_h edge_weight_rank')
    linkCounts = {'ML': [0], 'CL': [0]}
    s = time.perf_counter()
    # att = get_node_attrs_by_distance_from_root(ET, data, roots)
    R: list = roots.copy()
    # UET: nx.Graph = ET.to_undirected()

    RET = nx.reverse(ET)

    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))

    df = {"iter": [0],
          "interaction": [0], "ari": [ARI], "time": [0]}

    c, i = 0, 0

    # attr = np.zeros(len(ET.nodes))
    # eweight = {}

    edge_rank = []
    lennodes = 0
    for n in RET.nodes():
        neighbors = list(RET.neighbors(n))
        if not neighbors:
            continue

        # temp = [(ne, np.linalg.norm(data[ne] - data[n])) for ne in neighbors]
        temp = []
        _sum = 0
        for ne in neighbors:
            d = np.linalg.norm(data[ne] - data[n])
            _sum += d
            temp.append((ne, d))

        temp = [i for i, _ in sorted(temp, key=lambda x: x[1])]

        edge_rank.append([temp, len(temp), _sum / len(neighbors)])

    edge_rank = [i[0] for i in sorted(
        edge_rank, key=lambda x: (x[1], x[2]), reverse=True)]
    print(f'{edge_rank[:10]=}')
    dup = []
    for ii in edge_rank:
        lennodes += len(ii)
        dup.extend(ii)

    lenedge_rank = len(edge_rank)
    _edge = set(ET.edges())

    # for id in tqdm(range(ET.number_of_edges())):
    id = 0
    while True:
        # print(f'{edge=}, {count=},{_max=}')
        # print(f'26 {edge=}')
        lenedge_rank = len(edge_rank)
        iter = id % lenedge_rank
        nodes: list = edge_rank[iter]

        if not nodes:
            edge_rank.pop(iter)
            continue

        id += 1
        n = nodes.pop()
        edge = list(ET.edges(n))[0]
        _edge.remove(edge)
        a, b = edge

        count = 1

        update_link_counts(linkCounts, real_labels[a] == real_labels[b])

        if real_labels[a] != real_labels[b]:
            count += refine_dislike(ET, data,
                                    real_labels, R, [edge], None)

    #     # ARI = df['ari'][-1]
    #     # if True or i % 2 == 0:
        predict_labels = get_predict_labels(ET)

        ARI = adjusted_rand_score(real_labels, predict_labels)
    #     # ARI = rand_score(real_labels, predict_labels)
        t = time.perf_counter()
    #     # RI = rand_score(real_labels, predict_labels)

        c += count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
    #     # df['ari'].append(RI)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            break
    if df['ari'][-1] != 1:
        raise Exception(f'88 {dataName=} {len(_edge)=} and  {
                        df['ari'][-1]=} {edge_rank=}')
    linkCountsP = 'result/other/'
    os.makedirs(linkCountsP, exist_ok=True)
    lc = pd.DataFrame(linkCounts)
    # rename index name to iter
    lc.index.name = 'iter'
    lc.to_csv(f'{linkCountsP}{dataName}_linkCounts.csv', index=True)

    dfForalpha = kwargs['dfForalpha']
    # dfForalpha['dataName'].append(dataName)
    # dfForalpha['alpha'].append(alpha)
    # dfForalpha['beta'].append(1-alpha)
    # dfForalpha['query'].append(c)

    print(f'finash')
    return pd.DataFrame(df), R


def refine_by_h_layer_Q(ET: nx.Graph, roots: list, data, real_labels, subroots: list, dataName: str = 'nan', **kwargs):
    '''
    1. edge_weight_rank the farthest edge first
    2. 
    '''
    print(f'{dataName} start refine_by_h layer_Q')
    linkCounts = {'ML': [0], 'CL': [0]}
    s = time.perf_counter()
    # att = get_node_attrs_by_distance_from_root(ET, data, roots)
    R: list = roots.copy()
    # UET: nx.Graph = ET.to_undirected()

    RET = nx.reverse(ET)

    def geti(x):
        e = list(ET.edges(x))[0]
        return np.linalg.norm(data[e[0]] - data[e[1]])

    bfs_layers = dict(enumerate(nx.bfs_layers(RET, roots)))
    for i in range(1, len(bfs_layers)):
        temp = bfs_layers[i]

        temp = [i for i in sorted(temp, key=geti)]
        bfs_layers[i] = temp

    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))

    df = {"iter": [0],
          "interaction": [0], "ari": [ARI], "time": [0]}

    c, i = 0, 0
    edge_rank = []

    for i in range(1, len(bfs_layers)):
        edge_rank.append(bfs_layers[i])

    # for id in tqdm(range(ET.number_of_edges())):
    id = 0
    while True:
        # print(f'{edge=}, {count=},{_max=}')
        # print(f'26 {edge=}')
        lenedge_rank = len(edge_rank)
        iter = id % lenedge_rank
        nodes: list = edge_rank[iter]

        if not nodes:
            edge_rank.pop(iter)
            continue

        id += 1
        n = nodes.pop()
        edge = list(ET.edges(n))[0]
        a, b = edge

        count = 1

        update_link_counts(linkCounts, real_labels[a] == real_labels[b])

        if real_labels[a] != real_labels[b]:
            count += refine_dislike(ET, data,
                                    real_labels, R, [edge], None)

    #     # ARI = df['ari'][-1]
    #     # if True or i % 2 == 0:
        predict_labels = get_predict_labels(ET)

        ARI = adjusted_rand_score(real_labels, predict_labels)
    #     # ARI = rand_score(real_labels, predict_labels)
        t = time.perf_counter()
    #     # RI = rand_score(real_labels, predict_labels)

        c += count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
    #     # df['ari'].append(RI)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            break
    # if df['ari'][-1] != 1:
    #     raise Exception(f'88 {dataName=} {len(_edge)=} and  {
    #                     df['ari'][-1]=} {edge_rank=}')
    linkCountsP = 'result/other/'
    os.makedirs(linkCountsP, exist_ok=True)
    lc = pd.DataFrame(linkCounts)
    # rename index name to iter
    lc.index.name = 'iter'
    lc.to_csv(f'{linkCountsP}{dataName}_linkCounts.csv', index=True)

    dfForalpha = kwargs['dfForalpha']
    # dfForalpha['dataName'].append(dataName)
    # dfForalpha['alpha'].append(alpha)
    # dfForalpha['beta'].append(1-alpha)
    # dfForalpha['query'].append(c)

    print(f'finash')
    return pd.DataFrame(df), R


def refine_by_h_LabelSpreading(ET: nx.Graph, roots: list, data, real_labels, subroots: list, dataName: str = 'nan', alpha=.5, **kwargs):
    '''
    1.  LabelSpreading
    '''
    print(f'start refine_by_h LabelSpreading')
    linkCounts = {'ML': [0], 'CL': [0]}
    lp = LabelSpreading()

    # att = get_node_attrs_by_distance_from_root(ET, data, roots)
    R: list = roots.copy()

    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))

    df: dict[int, list] = {"iter": [0],
                           "interaction": [0], "ari": [ARI], "time": [0]}

    c, i = 0, 0
    edge_ranked_by_distance = {}
    edge_ranked_by_degree = {}

    _maxdist = 0
    _maxdegree = 0

    for edge in ET.edges():
        dist = np.linalg.norm(
            data[edge[0]]-data[edge[1]])
        degree = ET.in_degree(edge[0]) * ET.in_degree(edge[1])
        # 给ET的边加上weight 属性
        ET[edge[0]][edge[1]]['weight'] = dist

        edge_ranked_by_distance[edge] = dist
        edge_ranked_by_degree[edge] = degree

        _maxdist = max(_maxdist, dist)
        _maxdegree = max(_maxdegree, degree)

    # edges = list(ET.edges())
    beta = 1-alpha
    edge_ranked_by_dd = {
        edge: alpha*(edge_ranked_by_distance[edge]/_maxdist) + beta*(edge_ranked_by_degree[edge]/_maxdegree) for edge in ET.edges()}

    _edge_ranked = sorted(
        edge_ranked_by_dd.items(), key=lambda x: x[1], reverse=True)

    visted = set()
    neiborhoods = []
    labels = np.ones(len(ET.nodes))*-1
    A = nx.to_numpy_array(ET)
    # print(f'{A.shape=}')
    # np.savetxt('A.csv', A, delimiter=',', fmt='%.4f')
    # print(f'{type(edge_ranked_by_dd)=}')
    # print(f'{type(_edge_ranked)=}')

    for edge, v in _edge_ranked:
        s = time.perf_counter()
        # print(f'{edge=}, {count=},{_max=}')
        # print(f'26 {edge=}')
        a, b = edge
        # print(f'{edge=}')
        # print(f'{a=}, {b=}')
        count = 0

        if a not in visted:
            # print(f"{neiborhood_insertion(neiborhoods, a, real_labels)=}")
            la, cc = neiborhood_insertion(neiborhoods, a, real_labels)
            labels[a] = la
            count += cc
            visted.add(a)

        if b not in visted:
            # print(f"{neiborhood_insertion(neiborhoods, b, real_labels)=}")
            lb, cc = neiborhood_insertion(neiborhoods, b, real_labels)
            labels[b] = lb
            count += cc
            visted.add(b)

        update_link_counts(linkCounts, real_labels[a] == real_labels[b])
        # get ET's adjacency matrix
        # ET.adjacency_ = nx.to_numpy_array(ET)
        classs = [i for i in range(len(neiborhoods))]

        lp.fit(copy.deepcopy(A), labels, np.array(classs))

        predict_labels = lp.predict([])

        ARI = adjusted_rand_score(real_labels, predict_labels)
    #     # ARI = rand_score(real_labels, predict_labels)
        t = time.perf_counter()
    #     # RI = rand_score(real_labels, predict_labels)

        c += count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
    #     # df['ari'].append(RI)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            break

    linkCountsP = 'result/other/'
    os.makedirs(linkCountsP, exist_ok=True)
    lc = pd.DataFrame(linkCounts)
    # rename index name to iter
    lc.index.name = 'iter'
    lc.to_csv(f'{linkCountsP}{dataName}_linkCounts.csv', index=True)

    dfForalpha = kwargs['dfForalpha']
    dfForalpha['dataName'].append(dataName)
    dfForalpha['alpha'].append(alpha)
    dfForalpha['beta'].append(1-alpha)
    dfForalpha['query'].append(c)

    print(f'finash')
    return pd.DataFrame(df), R


if __name__ == '__main__':
    raise Exception('This is a module, not a script')
