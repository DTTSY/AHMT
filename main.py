import warnings
import time
import os
import queue
import copy

import logging
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
import numpy as np
from joblib import Parallel, delayed

from myutil import DataLoader
from myutil.retry import retry
from PRSCSWAP import PRS
from utils import *

from sklearn.metrics.pairwise import euclidean_distances

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='log.log', filemode='w', encoding="utf-8")

warnings.filterwarnings("ignore")


def get_auic(path: str):
    auic = {'dataset': [], 'auic': []}
    for file in os.listdir(path):
        auic['dataset'].append(file.split("_")[0])
        df = pd.read_csv(f'{path}/{file}')
        auic['auic'].append(df['ari'].sum()/(df.shape[0]*2))
    pd.DataFrame(auic).to_csv("auic.csv", index=False)


def draw_graph_w(ET: nx.Graph, data: np.array, roots: list, title: str, real_labels=None, subroots=None):
    # pos = nx.spring_layout(ET)
    # pos = nx.nx_agraph.graphviz_layout(ET)
    pos = nx.nx_agraph.graphviz_layout(ET, prog="dot")
    # pos = nx.nx_agraph.graphviz_layout(ET, prog="twopi")
    # pos = nx.nx_agraph.graphviz_layout(ET, prog="fdp")
    # pos = nx.nx_agraph.graphviz_layout(ET, prog="sfdp")
    # pos = nx.nx_agraph.graphviz_layout(ET, prog="circo")
    # pos = nx.nx_agraph.graphviz_layout(ET, prog="nop")
    # pos = nx.nx_agraph.graphviz_layout(ET, prog="nop2")
    # pos = nx.nx_agraph.graphviz_layout(ET, prog="osage")

    # pos = graphviz_layout(ET, prog="twopi")
    # pos = nx.kamada_kawai_layout(ET)
    # pos = nx.spectral_layout(ET)
    # pos = nx.shell_layout(ET)
    # pos = nx.circular_layout(ET)
    # pos = nx.planar_layout(ET)
    # pos = nx.random_layout(ET)
    # pos = nx.fruchterman_reingold_layout(ET)
    # pos = nx.bipartite_layout(ET, roots)
    plt.figure(figsize=(15, 10))
    nx.draw(ET, pos, with_labels=True)
    realcluster: dict[int, list] = {}

    for i, l in enumerate(real_labels):
        if l not in realcluster:
            realcluster[l] = []
        realcluster[l].append(i)
    # c = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
    # get 20 named colors
    c = plt.cm.tab20.colors

    # get named colors
    for k, v in realcluster.items():
        nx.draw_networkx_nodes(
            ET, pos, nodelist=v, node_color=c[k])
    # nx.draw_networkx_nodes(ET, pos, nodelist=roots, node_color='r')
    plt.title(title)
    # plt.savefig(f'result/p2/graph/{title}.pdf')
    # plt.show()
    plt.savefig(f'result/tree/{title}.png')


def get_label_vec_from_G(ET: nx.Graph):
    y = np.zeros(len(ET.nodes), dtype=int)
    label = nx.get_node_attributes(ET, 'label')
    t = -1
    for k, v in sorted(label.items(), key=lambda x: x[0]):
        assert abs(t-k) == 1
        y[k] = v
        t = k
    return y


def label_all_children(ET: nx.Graph, roots: list):
    # 将ET变为无向图
    T = ET
    if not nx.is_directed(ET):
        T = ET.to_undirected()
    # 将根节点的标签传递给所有的子节点
    for r in roots:
        for n in nx.descendants(T, r):
            # set node attribute
            T.nodes[n]['label'] = T.nodes[r]['label']


def get_G(ET: nx.Graph, data: np.array, gamma=1.):
    G = ET.to_undirected()
    for e in ET.edges:
        G.add_edge(
            e[0], e[1], weight=gamma*np.exp(-np.linalg.norm(data[e[0]]-data[e[1]])))
    # 得到邻接矩阵
    A = nx.adjacency_matrix(G).toarray()
    # 判断是否是对称矩阵
    assert np.allclose(A, A.T)
    return A


def refine_graph(ET: nx.Graph, data: np.array, roots: list, title: str, real_labels=None):
    df = {"iter": [0], "interaction": [0], "ari": [0], "time": [0]}
    N = []
    T = ET.to_undirected()
    for n in T.nodes:
        T.add_node(n, label=0, activated=False)

    # label_all_children(T, roots)
    Y = get_label_vec_from_G(T)
    print(Y)

    ari, ri, nmi = get_metric(real_labels, T)
    print(f"the ARI of {title} is {ari}")
    print(f"the RI of {title} is {ri}")
    print(f"the NMI of {title} is {nmi}")

    return pd.DataFrame(df), roots


def get_uncertainty_rank(ET: nx.Graph):
    attrs = nx.get_node_attributes(ET, 'uncertainty')
    attrs = sorted(attrs.items(), key=lambda x: x[1])
    return attrs


def draw_graph(ARIpath: str, pathToSavePic: str = 'result/pic', file='', remove=False, matric='ARI'):
    print('save result to pic')
    os.makedirs(pathToSavePic, exist_ok=True)
    # for file in os.listdir(ARIpath):
    # 判断是否是csv文件
    # if not file.endswith('.csv'):
    #     continue
    ARI = pd.read_csv(f'{ARIpath}/{file}')
    fig, ax = plt.subplots()
    ax.plot(ARI['interaction'], ARI['ari'], label='ARI')
    ax.set_xlabel('Quries')
    ax.set_ylabel(matric)
    ax.set_title(f'{file.split("_")[0]}')
    ax.legend()
    path = f'{pathToSavePic}/{file.split("_")[0]}.png'
    # print(f'save  to {path}')
    plt.savefig(path, dpi=600)
    # plt.show()


def get_node_attrs_by_distance_from_parent(G: nx.Graph, data: np.array):
    attrs = np.zeros(len(G.nodes))
    for edge in G.edges():
        i, j = edge
        attrs[i] = np.linalg.norm(data[i]-data[j])
    return attrs


def get_leves_rank_que_by_distance(G: nx.Graph, roots: list, att, data: np.array = None, reverse=True):
    bfs_layers = dict(enumerate(nx.bfs_layers(G, roots)))
    bfs_layers.pop(0)
    # sort by distance
    for l, nodes in bfs_layers.items():
        _nodes = [[n, att[n]] for n in nodes]
        _nodes = [n for n, d in sorted(
            _nodes, key=lambda x: x[1], reverse=reverse)]
        bfs_layers[l] = _nodes
    return bfs_layers


def put_neib_in_Q(ET, data, T, _maxdis, _maxdig, RankQ, n, inQ=set(), alpha=.5):
    beta = 1-alpha
    nbors = list(T.neighbors(n))
    # assert len(nbors) <= 1, f'{n=} has {len(nbors)} neighbors'
    for nb in nbors:
        if nb in inQ:
            continue
        # t1
        # RankQ.put(Node(nb, np.linalg.norm(data[nb]-data[n])))
        # t2
        dist = np.linalg.norm(data[n]-data[nb])
        dig = ET.in_degree(n) * ET.in_degree(nb)
        RankQ.put(Node(nb, alpha*(dist/_maxdis) + beta*(dig/_maxdig)))
        inQ.add(nb)


def edge_first_order_neighbors(G: nx.DiGraph, data, isNormalized=False):
    edge_neighbors_count: dict = {}
    edge_F1w_count = {}
    _max = 0
    for u, v in G.edges():
        # 找到连接到u或v的所有边
        le = G.in_degree(u) * G.in_degree(v)
        edge_F1w_count[(u, v)] = np.linalg.norm(data[u]-data[v])
        _max = max(_max, le)
        edge_neighbors_count[(u, v)] = le
    if isNormalized:
        for k in edge_neighbors_count:
            edge_neighbors_count[k] /= _max
            # u, v = k
            # edge_neighbors_count[k] /= G.in_degree(u) + G.in_degree(v)
    return edge_neighbors_count, _max, edge_F1w_count


def refine_by_h_6(ET: nx.Graph, roots: list, data, real_labels, subroots: list, dataName: str = 'nan', **kwargs):
    linkCounts = [0, 0, 0]
    linkCounts = {'ML': [0], 'CL': [0]}
    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))
    df: dict[int, list] = {"iter": [0],
                           "interaction": [0], "ari": [ARI], "time": [0]}
    c = 0
    visited = set()
    visited.add(roots[0])
    inQ = set()

    T: nx.Graph = nx.reverse(ET)
    R: list = roots.copy()

    print(f'start refine_by_h 6')
    print(f'{len(ET.edges)=}, {len(ET.nodes)=}')
    att = get_node_attrs_by_distance_from_root(ET, data, roots)

    i = 0
    Rank = subroots.copy()
    Rank += list(T.neighbors(roots[0]))
    Rank = set(Rank)
    Rank.remove(roots[0])

    _maxdis = 0
    _maxdig = 0
    for e in ET.edges():
        _maxdis = max(_maxdis, np.linalg.norm(data[e[0]] - data[e[1]]))
        _maxdig = max(_maxdig, ET.in_degree(e[0]) * ET.in_degree(e[1]))

    # init
    alpha = kwargs['alpha']
    beta = 1-alpha
    RankQ = queue.PriorityQueue()
    for n in Rank:
        e = list(ET.neighbors(n))

        dist = np.linalg.norm(data[n]-data[e[0]])
        dig = ET.in_degree(n) * ET.in_degree(e[0])
        RankQ.put(Node(n, alpha*(dist/_maxdis) + beta*(dig/_maxdig)))
        inQ.add(n)
        put_neib_in_Q(ET, data, T, _maxdis, _maxdig,
                      RankQ, n, inQ, alpha=alpha)

    if not Rank:
        print(f'all candidate nodes have been visited')
        return pd.DataFrame(df), R
    q = 1000
    ll = 0
    while not RankQ.empty():
        nn = RankQ.get(block=False)
        n = nn.id
        put_neib_in_Q(ET, data, T, _maxdis, _maxdig,
                      RankQ, n, inQ, alpha=alpha)
        s = time.time()
        count = 0
        e = list(ET.edges(n))
        if e:
            a, b = e[0]
            count += 1
            ll += 1

            update_link_counts(linkCounts, real_labels[a] == real_labels[b])

            if real_labels[a] != real_labels[b]:
                count += refine_dislike(ET, data,
                                        real_labels, R, e, att)

        predict_labels = get_predict_labels(ET)

        ARI = adjusted_rand_score(real_labels, predict_labels)
        t = time.perf_counter()

        c += count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            break

    print(f'finash')

    return pd.DataFrame(df), R


def refine_by_h_5(ET: nx.Graph, roots: list, data, real_labels, subroots: list, dataName: str = 'nan', alpha=.5, **kwargs):
    print(f'start refine_by_h 5')
    linkCounts = {'ML': [0], 'CL': [0]}
    s = time.perf_counter()
    R: list = roots.copy()
    ARI = adjusted_rand_score(real_labels, get_predict_labels(ET))

    df: dict[int, list] = {"iter": [0],
                           "interaction": [0], "ari": [ARI], "time": [0]}

    edge_neighbors_count, _md, _ = edge_first_order_neighbors(
        ET, data=data, isNormalized=True)
    _maxDist = 0
    print(f'max degree: {_md=}')
    for edge in edge_neighbors_count.keys():
        u, v = edge
        _maxDist = max(_maxDist, np.linalg.norm(data[u]-data[v]))
    print(f'max distance: {_maxDist}')

    beta = 1 - alpha
    for edge in edge_neighbors_count.keys():
        u, v = edge
        dist = np.linalg.norm(data[u]-data[v])
        edge_neighbors_count[edge] = beta * \
            edge_neighbors_count[edge] + alpha*(dist/_maxDist)

    sorted_edge_neighbors_count = sorted(
        edge_neighbors_count.items(), key=lambda x: x[1], reverse=True)
    c, i = 0, 0
    for edge, cc in sorted_edge_neighbors_count:
        e = edge
        a, b = edge
        count = 1

        update_link_counts(linkCounts, real_labels[a] == real_labels[b])

        if real_labels[a] != real_labels[b]:
            count += refine_dislike(ET, data,
                                    real_labels, R, [e], None)

        predict_labels = get_predict_labels(ET)
        if i % 10 == 0:
            ARI = adjusted_rand_score(real_labels, predict_labels)
        t = time.perf_counter()

        c += count
        i += 1

        df['iter'].append(i)
        df['interaction'].append(c)
        df['ari'].append(ARI)
        df['time'].append(t-s)

        if ARI == 1:
            print(f'find the best result at {i} iteration')
            break

    print(f'finash')
    return pd.DataFrame(df), R


@retry(retries=3, delay=0)
def run_PRSC(data, real_labels, K, num_thread, divide_method='PRSC'):
    # start = time.time()
    prs = PRS(data)
    threshold_clusters = K
    threshold_clusters = 2
    prs.get_clusters(num_thread, threshold_clusters,
                     divide_method=divide_method)
    # print(prs.boundary_nodes)
    ET, roots = prs.get_final_tree_nx()
    subroots = prs.get_subroots()
    print(f'the len of roots is {len(roots)}')
    print(f'{roots=}')
    roots = merge_roots(ET, roots)
    print(f'roots affter merge: {roots=}')
    assert len(roots) == 1, f'{roots=}'
    assert len([ET.subgraph(c) for c in nx.weakly_connected_components(ET)]) == 1, f'{
        len([ET.subgraph(c) for c in nx.weakly_connected_components(ET)])=}'
    nods = list(ET.nodes)
    assert len(real_labels) == len(nods), f'{len(real_labels)=}, {len(nods)=}'
    # 检查nodes里面的数是否连续
    nl = np.zeros(len(real_labels))
    nl[nods] = 1
    p1 = get_predict_labels(ET)
    p2: np.ndarray = np.zeros(len(real_labels), dtype=int)
    ari = adjusted_rand_score(real_labels, p1)
    ari2 = adjusted_rand_score(real_labels, p2)
    assert ari == ari2, f'{ari=}, {ari2=}, {len(nods)=}, {len(real_labels)=}, {
        np.all(p1 == p2)}, {len(roots)=}, len connected_components: {len([ET.subgraph(c) for c in nx.weakly_connected_components(ET)])}'
    # 判断两个数组是否相等
    assert np.all(p1 == p2), f'p1 is not equal to p2'
    # 查看ET有几个联通分量
    print(f'{nx.number_connected_components(ET.to_undirected())=}')
    print(f'{len(ET.edges)=}, {len(ET.nodes)=}')
    return ET, roots, subroots


def runp_h(dataDir, file: str, outdir: str = 'result/', tht=1) -> None:
    warnings.filterwarnings("ignore")
    assert os.path.exists(dataDir)

    rdata, real_labels, K, _ = DataLoader.get_data_from_local(
        f'{dataDir}/{file}', doPerturb=True, frac=1)

    data = rdata.copy()

    data = (data - data.mean()) / (data.std())

    for tht in range(1, 11):
        data = rdata.copy(deep=True)
        ET, roots, subroots = run_PRSC(data, real_labels, K, tht)

        data = data.values
        assert len(data) == len(list(ET.nodes))

        dfForalpha = {"dataName": [], "alpha": [], "beta": [], 'query': []}
        ARI_record, R = None, None

        for alpha in np.arange(0, 1.1, .1):
            ARI_record, R = refine_by_h_5(
                copy.deepcopy(ET), roots, data, real_labels, subroots, dataName=file.split(".")[0], alpha=alpha, dfForalpha=dfForalpha)
            outdir = f'result/exp-p/theta-{tht*100}/alpha-{alpha:.1f}'
            os.makedirs(outdir, exist_ok=True)
            ARI_record.to_csv(
                f'{outdir}/{file.split(".")[0]}.csv', index=False)


def calculate_AUIC(f: pd.DataFrame, n=0):
    # 二分查找interaction =n的行
    idx = np.searchsorted(
        f['interaction'].values, n, side='right')
    # 取出0-idx的数据
    f = f.iloc[:idx]['ari'].values
    # f = f['ari'].values
    sum = 0
    for i in range(len(f)-1):
        sum += f[i]+f[i+1]
    p = sum/(2*(len(f)-1))
    return p


def make_synthetic_data(n_samples, n_features, n_centers, cluster_std):
    rdata, real_labels = make_blobs(n_samples=n_samples,
                                    n_features=n_features,
                                    centers=n_centers,
                                    cluster_std=cluster_std)
    return pd.DataFrame(rdata), real_labels


def run_CPU_time_on_synthetic_data(size, k):
    n_samples = size
    # n_samples = 300
    n_features = 10
    n_centers = k
    cluster_std = 1.0

    rdata, real_labels = make_synthetic_data(
        n_samples=n_samples, n_centers=n_centers, n_features=n_features, cluster_std=cluster_std)
    return rdata, real_labels


def run_alg_turn_p(data, real_labels, K, tht, title):
    ET, roots, subroots = run_PRSC(data, real_labels, K, tht)
    data = data.values

    dfForalpha = {"dataName": [], "alpha": [], "beta": [], 'query': []}
    ARI_record, R = None, None

    for alpha in np.arange(0, 1.1, .1):
        ARI_record, R = refine_by_h_6(
            copy.deepcopy(ET), roots, data, real_labels, subroots, dataName=title, alpha=alpha, dfForalpha=dfForalpha)
        outdir = f'result/exp-p/theta-{tht*100}/alpha-{alpha:.1f}'
        os.makedirs(outdir, exist_ok=True)
        ARI_record.to_csv(
            f'{outdir}/{title}.csv', index=False)

    # alphaDir = 'result/alpha'
    # os.makedirs(alphaDir, exist_ok=True)
    # dForalpha = pd.DataFrame(dfForalpha)
    # # alpha列与beta列的值保留一位小数
    # dForalpha['alpha'] = dForalpha['alpha'].apply(lambda x: round(x, 1))
    # dForalpha['beta'] = dForalpha['beta'].apply(lambda x: round(x, 1))

    # dForalpha.to_csv(f'{alphaDir}/{file.split(".")[0]}.csv', index=False)

    # ARI_record, R = refine_by_h_5(
    #     copy.deepcopy(ET), roots, data, real_labels, subroots, dataName=file.split(".")[0])
    # print(f'{R=}, {len(R) == K}', alpha=.5, dfForalpha=dfForalpha)
    # dfForalpha = pd.DataFrame(dfForalpha)
    # dfForalpha.to_csv(f'result/alpha/{file.split(".")[0]}.csv', index=False)

    # ARI_record, R = refine_by_h_6(
    #     ET, roots, data, real_labels, subroots, dataName=file.split(".")[0])
    # print(f'{R=}, {len(R) == K}')

    # ARI_record, R = refine_by_h_4(
    #     ET, roots, data, real_labels, subroots, dataName=file.split(".")[0])
    # print(f'{R=}, {len(R) == K}')

    # draw_graph_w(ET, data, R, title=file.split(
    #     ".")[0], real_labels=real_labels)

    # ARI_record, _ = DSL(data, real_labels, title=file.split(".")[
    #     0], skeleton=ET, representatives=roots, K=K)

    # tail = ARI_record.tail(1)
    # ari21['dataset'].append(file.split(".")[0])
    # ari21['ari'].append(tail['ari'].values[0])
    # ari21['interaction'].append(tail['interaction'].values[0])
    # print(f'run on {file} finished')

    # df = pd.DataFrame(ari21)
    # drop air is less than 1
    # df = df[df['ari'] == 1]
    # df.to_csv("our.csv", index=False)
    # pd.DataFrame(info).to_csv("info.csv", index=False)
    # get_auic('result/p2')

    # os.makedirs(outdir, exist_ok=True)
    # ARI_record.to_csv(
    #     f'{outdir}/{file.split(".")[0]}.csv', index=False)


def runp_p(dataDir, file: str, outdir: str = 'result/p2/small', tht=1) -> None:
    warnings.filterwarnings("ignore")
    assert os.path.exists(dataDir)

    rdata, real_labels, K, _ = DataLoader.get_data_from_local(
        f'{dataDir}/{file}', doPerturb=True, frac=1)
    title = file.split(".")[0]

    data = rdata.copy()

    # data = StandardScaler().fit_transform(data)
    # data = pd.DataFrame(data)

    data = (data - data.mean()) / (data.std())

    Parallel(n_jobs=10)(delayed(run_alg_turn_p)(rdata.copy(
        deep=True), real_labels, K, tht, file.split(".")[0]) for tht in range(1, 11))


def run():
    os.makedirs('result', exist_ok=True)

    arg: dict = {'dataDir': 'data',
                 'resultDir': 'result'}

    files: list[str] = os.listdir(arg['dataDir'])

    files = ['Thyroid.csv']

    jobs = len(files)
    n_jobs: int = 1
    batch_size = max(1, (jobs + n_jobs-1) // n_jobs)
    Parallel(n_jobs=n_jobs, batch_size=batch_size)(delayed(runp_h)(arg['dataDir'], file,
                                                                   arg['resultDir'], tht=1) for file in files
                                                   if file.endswith('.csv'))


if __name__ == '__main__':
    run()
