import random
from sklearn.utils import shuffle
import math
import threading
import time
from sklearn.metrics import adjusted_rand_score, rand_score, normalized_mutual_info_score
# from util.estimate import rand_index
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn import cluster as cluster_methods
from sklearn.neighbors import NearestNeighbors
import scipy
# 导入labelencodeer
# from sklearn.preprocessing import LabelEncoder
from myutil import DataLoader
import networkx as nx
import os
from myutil.retry import retry
from joblib import Parallel, delayed
import traceback
# from main import runPRSC


class PRS():
    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data
        self.clusters = []
        self.final_tree = {}
        self.nx_graph = nx.DiGraph()
        self.results = None
        self.s_score = {}
        self.boundary_points_ = set()
        self.subroots = set()
        self.ssubroots = set()

    def get_subroots(self):
        return list(self.subroots)

    def divide_data_random(self, k):
        # d = self.data
        d = self.data.take(np.random.permutation(self.data.shape[0]))
        split_threshold = int(d.shape[0] / k)
        sub_data = [
            d.iloc[i * split_threshold:(i + 1) * split_threshold, :] for i in range(k - 1)]
        sub_data.append(d.iloc[(k - 1) * split_threshold:, :])
        return sub_data

    def divide_data_random_m(self, k):
        # devide data radmely into k parts
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        sub_data = []
        for train_index, test_index in kf.split(self.data):
            train_df = self.data.iloc[train_index]
            test_df = self.data.iloc[test_index]
            sub_data.append(pd.concat([train_df, test_df]))

        return sub_data

    def get_portion_info(self, data: dict):
        # get the portion of each class
        x = []
        y = []
        for k, v in data.items():
            x.append(k)
            y.append(len(v))
        # 显示y的值
        bars = plt.bar(x, y)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                     f'{height}', ha='center', va='bottom')
        plt.show()
        return

    def divide_data_kmeans(self, k):
        if k == 1:
            return [self.data]
        print(f'divide_data_kmeans: the missing value in data is {
              self.data.isnull().sum().sum()}')

        kmeans_model = cluster_methods.KMeans(
            n_clusters=k, init='k-means++').fit(self.data)

        patition = kmeans_model.labels_
        print(f'{self.data.shape=}')
        print(f'{patition.shape=}')
        sub_data = []
        results_map_no = {}
        sup = set(patition)
        print(f'{len(sup)=}')

        for s in sup:
            results_map_no[s] = []

        for i in range(len(patition)):
            results_map_no[patition[i]].append(i)
        maxl, maxk = 0, 0

        for k, v in results_map_no.items():
            maxl = max(maxl, len(v))
            if maxl == len(v):
                maxk = k

        for s in results_map_no.values():
            if len(s) <= 1:
                s.append(results_map_no[maxk].pop())
            sub_data.append(self.data.take(s))

        print(f'{len(sub_data)=}')
        # self.get_portion_info(results_map_no)
        return sub_data

    def divide_data_results(self, results):
        sub_data = []
        results_map_no = {}
        sup = set(results.values())
        for s in sup:
            results_map_no[s] = []
        for key in results.keys():
            results_map_no[results[key]].append(key)
        for s in results_map_no.values():
            sub_data.append(self.data.take(s))
        return sub_data

    def divide_data_PRSC(self, theta, k=0):

        ET = run_PRSC(
            self.data, theta, K=k)
        # ET 是networkx Graph对象，取出ET的联通分量
        sub_data = []
        S = [ET.subgraph(c) for c in nx.weakly_connected_components(ET)]
        for s in S:
            sub_data.append(self.data.take(s.nodes()))
        return sub_data

    def dist(self, a, b):
        return np.linalg.norm(self.data.values[a] - self.data.values[b])

    def get_clusters(self, num_thread, threshold_clusters, divide_method='defalut', lables=[]):
        # print('50', threshold_clusters)
        if divide_method == 'kmeans':
            sub_data = self.divide_data_kmeans(num_thread)
        elif divide_method == 'PRSC':
            sub_data = self.divide_data_PRSC(num_thread)
        else:
            sub_data = self.divide_data_random(num_thread)
        # sub_data = self.divide_data_random(num_thread)
        # sub_data = self.divide_data_kmeans(num_thread)
        # sub_data = self.divide_data_PRSC(100, lables)
        bns_ = []
        no_bns = math.log(self.data.shape[0])
        for i in range(len(sub_data)):
            pairwise = get_boundary_cross(
                sub_data[i].values, math.ceil(no_bns))
            for pair in pairwise:
                bns_.extend(pair)
        # print('61', bns_)
        self.boundary_nodes = get_boundary_cross(bns_, math.ceil(no_bns/8))
        # sub_data = [self.data]
        edges = self.iteraction(sub_data, threshold_clusters)

        # for k, v in edges2community(edges).items(): print('1-',len(v))

        edges.update(self.swap(edges))

        # for k, v in edges2community(edges).items(): print('2-', len(v))
        edges.update(self.check_tiny_community(edges))
        # for k, v in edges2community(edges).items(): print('3-',len(v))

        # clustering_tree, roots = get_tree(edges)
        #
        #
        # self.results = get_groups(roots, clustering_tree)
        # print(self.results)
        # print('42:',self.results)
        # for i in range(1):
        #     # # for k, v in edges2community(edges).items(): print('1-',len(v))
        #     # edges.update(self.swap(edges))
        #     # # for k, v in edges2community(edges).items(): print('2-', len(v))
        #     # edges.update(self.check_tiny_community(edges))
        #     # # for k, v in edges2community(edges).items(): print('3-',len(v))
        #
        #
        #     clustering_tree, roots = get_tree(edges)
        #     self.results = get_groups(roots, clustering_tree)
        #     # print('62', self.results)
        #
        #     sub_data = self.divide_data_results(self.results)
        #     # for sd in sub_data: print(len(sd))
        #     edges = self.iteraction(sub_data, threshold_clusters)
        #     # clustering_tree, roots = get_tree(edges)
        #     #
        #     # self.results = get_groups(roots, clustering_tree)
        #     #
        #     # self.results = self.iteraction(sub_data, threshold_clusters)

        edges.update(self.shortestlinke(edges, threshold_clusters))

        # file = open('/Users/wenboxie/Desktop/edegs.csv', 'w')
        # file.write('Source, Target\n')
        # for k, v in edges.items():
        #     file.write(str(k)+','+str(v)+'\n')
        # file.close()

        clustering_tree, roots = get_tree(edges)
        # self.nx_graph.add_edges_from([(v, k) for k, v in edges.items()])
        self.nx_graph.add_edges_from(edges.items())
        # self.nx_graph = nx.DiGraph.reverse(self.nx_graph)
        # print('clustering_tree : ', clustering_tree)
        # print('roots : ', roots)
        self.final_tree['tree'] = clustering_tree
        self.final_tree['roots'] = list(roots)
        for r in roots:
            self.nx_graph.remove_edge(r, r)

        self.results = get_groups(roots, clustering_tree)
        # print('results: ', self.results)
        return

    def get_final_tree_nx(self):
        # assert self.nx_graph is None or self.final_tree[
        #     'roots'] is None, f'get_final_tree_nx error: nx_graph or roots is None'
        return self.nx_graph, self.final_tree['roots']

    def get_groups_nx(self, roots: set):
        result = {}
        print('root:', roots)

        for root in roots:
            result[root] = root
            for node in nx.bfs_tree(self.nx_graph, root).nodes():
                result[node] = root
        return result

    def iteraction(self, sub_data, threshold_clusters) -> dict:

        # print('length of sub-cluster:', len(sub_data))
        edges = {}
        sup_nodes = set()
        sub_clusters = []
        bn = []
        new_bn = []
        # print('len of sub data:', len(sub_data))
        thr = [cluster(i, sub_data[i], sub_clusters, self.boundary_nodes,
                       threshold_clusters, self.s_score, self.ssubroots) for i in range(len(sub_data))]
        for t in thr:
            t.start()
        for t in thr:
            t.join()

        # print(sub_clusters)
        for (s_n, e) in sub_clusters:
            sup_nodes = sup_nodes | s_n
            edges.update(e)

        # print('55:', len(edges.keys()))
        # print(edges)
        # print(len(sub_clusters))
        sup_nodes, edges = self.reduce(edges, sup_nodes, threshold_clusters)

        # print('62:', len(edges.keys()))
        # print(edges)
        # ef = open('/Users/wenboxie/Desktop/edegs.csv','w')
        # ef.write('Target,Source\n')
        # for k in edges.keys():
        #     ef.write(str(k)+','+str(edges[k])+'\n')
        # ef.close()

        # edges.update(self.shortestlinke(edges, sup_nodes, threshold_clusters))

        # clustering_tree, roots = get_tree(edges)

        # print(clustering_tree)

        # print('70:',len(roots))

        # for k in clustering_tree.keys():
        #     for v in clustering_tree[k]:
        #         print(self.data.values[v,0],'\t', self.data.values[v,1],'\t',self.data.values[k,0],'\t', self.data.values[k,1])
        # return get_groups(roots, clustering_tree)

        # for k,v in edges2community(edges).items(): print(len(v))
        return edges

    def reduce(self, edges, roots, threshold_clusters):
        # print('sub_clusters:', sub_clusters)
        roots = get_roots_from_edges(edges)
        self.subroots.update(roots)
        # # additional_edges = {}
        #
        # for c, p in edges.items():
        #
        #     if c == p:
        #         # print(c, ':', p)
        #         roots.add(p)
        # for k, v in edges2community(edges).items(): print('132:', len(v))

        # data_root = self.data[sub_clusters[:,0] == sub_clusters[:,0]]
        data_roots = self.data[self.data.index.isin(roots)]
        # print('reduce:',data_roots)
        ne = []
        thread = cluster(999, data_roots, ne,
                         self.boundary_nodes, threshold_clusters, self.s_score)
        thread.start()
        thread.join()
        sup_nodes = ()
        try:
            sup_nodes = (ne[0][0])
            edges.update(ne[0][1])
        except:
            sup_nodes = ()

        # print('110:', len(edges.keys()))
        return sup_nodes, edges

    # def swap(self, edges):
    #     # row_names = data._stat_axis.values.tolist()
    #     roots = get_roots_from_edges(edges)
    #     comms = edges2community(edges)
    #     new_edges = {}
    #     for r1, nodes in comms.items():
    #         for n in nodes:
    #             min = float("inf")
    #             min_r2 = r1
    #             for r2 in roots - set([r1]):
    #                 if dist(n,r2) < min:
    #                     min = dist(n, r2)
    #                     min_r2 = r2
    #             if min < dist(r1, n) and dist(r1, min_r2) < dist(r1, n):
    #                 new_edges[n] = min_r2
    #     # print('nedges:', new_edges)
    #     return new_edges
    def check_tiny_community(self, edges):
        tiny_comms = set()
        for k, v in edges2community(edges).items():
            if len(v) <= 5:
                tiny_comms.add(k)
        if len(tiny_comms) == 0:
            return {}
        # print('tiny cmmm:', tiny_comms)
        min_dist = {}
        min_no = {}
        for r1 in tiny_comms:
            min_dist[r1] = float('inf')
            min_no[r1] = r1
            for r2 in edges2community(edges).keys() - tiny_comms:
                di = self.dist(r1, r2)
                if di < min_dist[r1]:
                    min_dist[r1] = di
                    min_no[r1] = r2
        new_edges = {}
        for r1, r2 in min_no.items():
            new_edges[r1] = r2
        return new_edges

    def swap(self, edges):
        roots = get_roots_from_edges(edges)
        # roots_list = list(roots)
        comm = edges2community(edges)
        nearest_root_each_comm = {}
        for r1 in roots:
            nodes_in_r1 = comm[r1]
            nearest_root = {}
            for n in nodes_in_r1:
                nearest_root[n] = r1
            shortest_dist = {}
            for n in nodes_in_r1:
                shortest_dist[n] = float('inf')
            for r2 in roots - set([r1]):
                dist_r1_plus_r2 = {}
                for n in nodes_in_r1:
                    dist_r1_plus_r2[n] = (
                        self.dist(n, r1) + self.dist(n, r2)) * self.dist(r1, r2)
                for n in nodes_in_r1:
                    if dist_r1_plus_r2[n] < shortest_dist[n]:
                        shortest_dist[n] = dist_r1_plus_r2[n]
                        nearest_root[n] = r2
            nearest_root_each_comm[r1] = nearest_root
        nodes_bet_r1r2 = {}
        for r1 in roots:
            for r2 in roots - set([r1]):
                if (r1, r2) not in nodes_bet_r1r2.keys() and (r2, r1) not in nodes_bet_r1r2.keys():
                    nodes_bet_r1r2[r1, r2] = {}
        for r1 in roots:
            for n in comm[r1]:
                r2 = nearest_root_each_comm[r1][n]
                diff = self.dist(n, r1) - self.dist(n, r2)
                if (r1, r2) in nodes_bet_r1r2.keys():
                    nodes_bet_r1r2[(r1, r2)][n] = diff
                elif (r2, r1) in nodes_bet_r1r2.keys():
                    nodes_bet_r1r2[(r2, r1)][n] = -diff
        new_edges = {}
        for (r1, r2) in nodes_bet_r1r2.keys():
            # for r1 in roots:nodes_bet_r1r2
            #     for r2 in range(r1, len(roots_list)):
            nodes_and_diff = nodes_bet_r1r2[(r1, r2)]
            # find min in r2 and max in r1.
            nodes_in_r1 = set()
            nodes_in_r2 = set()
            for n in nodes_and_diff.keys():
                if n in comm[r1] - set([r1]):
                    nodes_in_r1.add(n)
                elif n in comm[r2] - set([r2]):
                    nodes_in_r2.add(n)
                # else:
                #     print('error in 192!')

            # max in r1
            if len(nodes_in_r1) == 0 or len(nodes_in_r2) == 0:
                continue
            max_r1_value = float('-inf')
            anchor_r1_value = float('-inf')
            max_r1 = r1
            anchor_r1 = r1
            for n in nodes_in_r1:
                diff = nodes_and_diff[n]
                if diff >= max_r1_value:
                    anchor_r1_value = max_r1_value
                    max_r1_value = diff
                    anchor_r1 = max_r1
                    max_r1 = n
                elif diff > anchor_r1_value:
                    anchor_r1_value = diff
                    anchor_r1 = n
            if max_r1 == r1:
                continue
            # min in r2
            min_r2_value = float('inf')
            anchor_r2_value = float('inf')
            min_r2 = r2
            anchor_r2 = r2
            for n in nodes_in_r2:
                diff = nodes_and_diff[n]
                if diff <= min_r2_value:
                    anchor_r2_value = min_r2_value
                    min_r2_value = diff
                    anchor_r2_value = min_r2
                    min_r2 = n
                elif diff < anchor_r2_value:
                    anchor_r2 = n
                    anchor_r2_value = diff
            if min_r2 == r2:
                continue

            # swap
            if min_r2_value < max_r1_value:
                ban = set([anchor_r1, anchor_r2, r1, r2])
                pre_anchor_r1 = edges[anchor_r1]
                pre_anchor_r2 = edges[anchor_r2]
                ban_r1 = set([anchor_r1])
                ban_r2 = set([anchor_r2])
                c = anchor_r1
                while c != r1:
                    c = edges[c]
                    ban_r1.add(c)
                c = anchor_r2
                while c != r2:
                    c = edges[c]
                    ban_r2.add(c)

                flag_anchor_r1 = True
                flag_anchor_r2 = True
                dist_anchor_r1_2_pre = self.dist(pre_anchor_r1, anchor_r1)
                dist_anchor_r2_2_pre = self.dist(pre_anchor_r2, anchor_r2)
                for n, diff in nodes_and_diff.items():
                    if n in ban:
                        continue
                    if min_r2_value < diff < max_r1_value:
                        dist2r1 = self.dist(n, anchor_r1)
                        dist2r2 = self.dist(n, anchor_r2)
                        dist2ori = self.dist(n, edges[n])
                        if dist2r1 < dist2ori and n not in ban_r1:
                            new_edges[n] = anchor_r1
                            if dist2r1 < dist_anchor_r1_2_pre and flag_anchor_r1:
                                if self.dist(r1, anchor_r1) < self.dist(r2, anchor_r1):
                                    new_edges[anchor_r1] = r1
                                else:
                                    new_edges[anchor_r1] = r2
                                flag_anchor_r1 = False

                        elif dist2r2 < dist2ori and n not in ban_r2:
                            new_edges[n] = anchor_r2
                            if dist2r2 < dist_anchor_r2_2_pre and flag_anchor_r2:
                                if self.dist(r2, anchor_r2) < self.dist(r1, anchor_r2):
                                    new_edges[anchor_r2] = r2
                                else:
                                    new_edges[anchor_r2] = r1
                                flag_anchor_r2 = False

                            # new_edges[n] = r2
                    # pre_anchor_r1 = edges[anchor_r1]
                    # pre_anchor_r2 = edges[anchor_r2]
                    # if pre_anchor_r1 in new_edges.keys() or pre_anchor_r2 in new_edges.keys():
                    #     new_edges[anchor_r1] = r1
                    #     new_edges[anchor_r2] = r2
                    # if pre_anchor_r1 in nodes_and_diff.keys():
                    #     if min_r2_value < nodes_and_diff[pre_anchor_r1] and nodes_and_diff[pre_anchor_r1] < max_r1_value:
                    #         # print('found it')
                    #         anchor_r1 = edges[pre_anchor_r1]
                    #
                    # if pre_anchor_r2 in nodes_and_diff.keys():
                    #     if min_r2_value < nodes_and_diff[pre_anchor_r2] and nodes_and_diff[pre_anchor_r2] < max_r1_value:
                    #         # print('found it')
                    #         anchor_r2 = edges[pre_anchor_r2]

            # reconstruction
            # if st_min_in_r2 < max_in_r1:
            #     rec_nodes = set()
            #     for n, diff in nodes_and_diff.items():
            #         if st_min_in_r2 < diff and diff < max_in_r1:
            #             rec_nodes.add(n)
            #     rec_nodes.add(rd_max_no_in_r1)
            #     rec_nodes.add(rd_min_no_in_r2)
            #     rec_data = []
            #     rec_data.append(self.data.take(list(rec_nodes)))
            #     for i in range(threshold_clusters):
            #         new_edges = self.iteraction(rec_data, threshold_clusters/(i+1))
            #         if len(new_edges.keys()) > 0:
            #             break

        # print('new edges:',len(new_edges.keys()),'roots:',set(new_edges.values()))
        return new_edges

    def get_results(self):
        # result = self.get_groups_nx(self.final_tree['roots'])

        return self.results

    # def detect_communities(self, A: np.matrix, roots, labels):
    #     parents_next = roots
    #     while len(parents_next) != 0:
    #         labels_new = set()
    #         for p in parents_next:
    #             # print('p=',p)
    #             for c in range(A.shape[0]):
    #                 if A[c, p] == 1:
    #                     labels[c] = labels[p]
    #                     # print(c, '->', labels[p],'(',c,'->',p,')')
    #                     labels_new.add(c)
    #         if len(labels_new) > 0:
    #             parents_next = labels_new - parents_next
    #         else:
    #             break

    def shortestlinke(self, edges, K):
        # roots = set()
        #
        # for c, p in edges.items():
        #
        #     if c == p:
        #         # print(c, ':', p)
        #         roots.add(p)
        roots = get_roots_from_edges(edges)

        # sup_nodes = roots
        additional_edges = {}

        if len(roots) > K:
            data_roots = self.data[self.data.index.isin(roots)]
            row_names = data_roots.index.values.tolist()
            d = data_roots.values
            neighbors = NearestNeighbors(n_neighbors=2)
            neighbors.fit(d)

            distance_ = neighbors.kneighbors()[0][:, 0]
            index_ = neighbors.kneighbors()[1][:, 0]

            sorted_distance_index = [0]
            for i in range(1, len(index_)):
                flag = False
                for j in range(len(sorted_distance_index)):
                    if distance_[i] < distance_[sorted_distance_index[j]]:
                        sorted_distance_index.insert(j, i)
                        flag = True
                        break
                if flag == False:
                    sorted_distance_index.append(i)
            # print(top)
            additional_edges = {}
            for i in roots:
                additional_edges[i] = i

            n_aditional_edges = 0
            for i in range(len(sorted_distance_index)):
                c = row_names[i]
                p = row_names[index_[i]]

                if additional_edges[p] != c:
                    additional_edges[c] = p
                    n_aditional_edges += 1
                    # sup_nodes.remove(c)
                if len(roots) - n_aditional_edges == K:
                    break

        return additional_edges


def get_groups(roots, clustering_tree) -> dict:
    # cout = 0
    result = {}
    for r in roots:
        result[r] = r
        # print('121(',cout,'):', result[r], '->', r)
        # cout+=1
    # print('roots:',roots)
    parents_next = roots
    while len(parents_next) != 0:
        labels_new = set()
        for p in parents_next:
            if p in clustering_tree.keys():
                for c in clustering_tree[p]:
                    result[c] = result[p]
                # print('121(',cout,'):',result[c], '->', c)
                # cout+=1
                labels_new = clustering_tree[p] | labels_new
        if len(labels_new) > 0:
            parents_next = labels_new - parents_next
        else:
            break
    # print('126:', len(result.keys()))
    return result


def get_farthest_node_c(sources, data, banning):
    max = float('-inf')
    A = sources[0]
    if len(sources) > 1:
        B = sources[1]
    if len(sources) == 3:
        C = sources[2]
    target = A
    for t in range(len(data)):
        if t in banning:
            continue
        AT = np.linalg.norm(data[A] - data[t])
        if len(sources) == 1:
            d = AT
        else:
            BT = np.linalg.norm(data[B] - data[t])
            AB = np.linalg.norm(data[A] - data[B])

            if len(sources) == 2:
                dd = (abs(AT - BT) + AB)
                if dd == 0:
                    traceback.print_stack()
                    # raise ValueError('### dd is zero')
                d = (AT + BT) / (abs(AT - BT) + AB)
            else:
                CT = np.linalg.norm(data[C] - data[t])
                d = (AT + BT + 2 * CT) / (abs(AT - BT) + AB)
        if d > max:
            max = d
            target = t
    banning.append(target)
    return target


def get_farthest_node(sources, data, banning):
    max = float('-inf')
    e = 1e-6
    A = sources[0]
    if len(sources) > 1:
        B = sources[1]
    if len(sources) == 3:
        C = sources[2]
    target = A
    for t in range(len(data)):
        if t in banning:
            continue
        AT = np.linalg.norm(data[A] - data[t])
        if len(sources) == 1:
            d = AT
        else:
            BT = np.linalg.norm(data[B] - data[t])
            AB = np.linalg.norm(data[A] - data[B])

            if len(sources) == 2:

                # dd = (abs(AT - BT) + AB)
                # if dd == 0:
                #     print(f'# {os.getpid()}:{
                #           threading.get_ident()} dd is zero and {A=} {B=} {t=} {AT=:.2f} {BT=:.2f} {AB=:.2f}')
                #     raise ValueError('### dd is zero')
                d = (AT + BT) / ((abs(AT - BT) + AB)+e)
            else:
                CT = np.linalg.norm(data[C] - data[t])
                d = (AT + BT + 2 * CT) / ((abs(AT - BT) + AB)+e)
        if d > max:
            max = d
            target = t
    banning.append(target)
    return target


def get_pairwise_bn(source, data, banning):
    C = get_farthest_node(source, data, banning)
    source.append(C)
    D = get_farthest_node(source, data, banning)
    return C, D


def get_boundary_cross(data, times):
    boundary_pairwise_nodes = []
    banning = []
    for i in range(math.ceil(times)):
        o = random.sample(range(len(data)), 1)
        banning.append(o[0])
        # print('483',o)
        a, b = get_pairwise_bn(o, data, banning)
        boundary_pairwise_nodes.append([data[a], data[b]])
        sources = [[a, b]]
        for no in range(3):
            source_ = sources.pop()
            l, r = get_pairwise_bn(source_, data, banning)
            boundary_pairwise_nodes.append([data[l], data[r]])
            sources.append([source_[0], l])
            sources.append([source_[0], r])
            sources.append([source_[1], l])
            sources.append([source_[1], r])
    return boundary_pairwise_nodes


class cluster(threading.Thread):
    def __init__(self, threadID, data, sub_clusters, bondary_nodes, threshold_clusters, s_score={}, subroots: set = None):
        threading.Thread.__init__(self)
        self.data = data
        self.threadID = threadID
        self.threshold_clusters = threshold_clusters
        self.sub_clusters = sub_clusters
        self.boundary_nodes = bondary_nodes
        self.s_score = s_score
        self.subroots = subroots

    def run(self):
        self.sub_clusters.append(self.aggregate(
            self.data, self.boundary_nodes))

    def aggregate(self, data: pd.DataFrame, boundary_nodes):
        row_names = data.index.values.tolist()
        if data.shape[0] == 0:
            return set(), {}
        elif data.shape[0] == 1:
            # data.values.tolist()
            return set(), {row_names[0]: row_names[0]}

        # row_names = data.columns.values.tolist()
        # row_names = data._stat_axis.values.tolist()

        # print(data)
        # 1. get the adjacent matrix and the corresponding relational matrix
        # A, R = get_adjacent_matrix_sparse(data)

        A, R = get_adjacent_matrix(data)
        # A.eliminate_zeros()
        # R.eliminate_zeros()

        # 2. get supporting nodes
        # print('R',R)

        sup_nodes = self.get_supporting_nodes(
            data, R, row_names, boundary_nodes)

        # sup_nodes = self.get_supporting_nodes_sparse(
        #     data, R, row_names, boundary_nodes)

        # print("thread-", self.threadID, '3: supporting node:', sup_nodes)

        # 3. if the number of sn smaller than K, stop aggregating
        edges = {}
        # if self.threadID == 999:
        #     print('402 threshold_clusters:',self.threshold_clusters)
        #     print('403 len of sup_nodes:', len(sup_nodes))
        if self.threshold_clusters <= len(sup_nodes):
            # if self.threadID == 999:print('407')
            # 3-1.  对于不是根结点的节点，看作已经确定了邻居，此时就返回其指向，
            #       对于根结点就看作没有确定的节点，近一步探索，迭代到下一层。

            for i in range(A.shape[0]):
                # i -> \deta_i
                if row_names[i] not in sup_nodes:
                    edges[row_names[i]] = row_names[A[i].argmax()]

            new_sup_nodes, new_edges = self.aggregate(
                data[data.index.isin(sup_nodes)], boundary_nodes)
            edges.update(new_edges)
            if len(new_sup_nodes) != 0:
                sup_nodes = new_sup_nodes
            # edges.update(self.aggregate(data_roots))

        else:
            # if self.threadID == 999:print('420')
            # 3-2. otherwise, in A, roots direct to;
            #      如果下一次聚类得到的类个数小于预期就停止聚类，
            #      每跟根点指向自己作为根节点的标记。
            for i in row_names:
                edges[i] = i
                # self.subroots.add(i)
            sup_nodes = set()

        # print('181-update:', edges)
        return sup_nodes, edges

    def get_boundary(self, data):
        sample = random.sample(range(data.values.shape[0]), int(
            np.ceil(math.log2(data.values.shape[0]))))
        references = []
        ref_set = set()
        for sam_i in sample:
            max = 0
            argmax = sam_i
            for i in range(data.values.shape[0]):
                if i in ref_set:
                    continue
                d = np.linalg.norm(data.values[sam_i] - data.values[i])
                if d > max:
                    max = d
                    argmax = i
            max = 0
            argmaxargmax = argmax
            for i in range(data.values.shape[0]):
                if i in ref_set:
                    continue
                d = np.linalg.norm(data.values[argmax] - data.values[i])
                if d > max:
                    max = d
                    argmaxargmax = i

            references.append([data.values[argmax], data.values[argmaxargmax]])
            ref_set.add(argmax)
            ref_set.add(argmaxargmax)
        return references

    def get_boundary_distance(self, n, b):
        return abs(np.linalg.norm(self.data.values[n] - b[0]) - np.linalg.norm(
            self.data.values[n] - b[1]))

    def add_sup_node(self, supporting_nodes, candidates, score_1, score_2, s1, s2, row_names):
        if score_1 >= score_2:
            supporting_nodes.add(row_names[s1])
        else:
            supporting_nodes.add(row_names[s2])

        candidates.remove(s2)
        candidates.remove(s1)
        # return

    def get_supporting_nodes(self, data, R, row_names, boundary_nodes):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        # print(R)
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                s2 = R[s1].argmax()  # s1和s2 是一对RNNs
                degree_1 = R[s1].sum()
                degree_2 = R[s2].sum()
                # self.add_sup_node(supporting_nodes, candidates, random.random(), random.random(), s1, s2, row_names)
                # continue
                # print('look')
                if degree_1 == 2 and degree_2 == 2:  # 如果是孤立的一堆RNNs，直接判断点对位置，忽略后续计算。
                    score_1 = 0
                    score_2 = 0
                    for re in boundary_nodes:
                        score_1 += self.get_boundary_distance(s1, re)
                        score_2 += self.get_boundary_distance(s2, re)
                    self.add_sup_node(supporting_nodes, candidates,
                                      score_1, score_2, s1, s2, row_names)
                    continue

                n_1 = 0
                n_2 = 0
                di_1 = 0
                di_2 = 0
                for i in range(np.size(R[s1])):
                    if R[s1, i] > 0:
                        n_1 += 1
                        di_1 += R[i].sum()
                for i in range(np.size(R[s2])):
                    if R[s2, i] > 0:
                        n_2 += 1
                        di_2 += R[i].sum()

                ave_neighbor_degree_1 = di_1 / n_1
                ave_neighbor_degree_2 = di_2 / n_2

                if ave_neighbor_degree_1 == ave_neighbor_degree_2:
                    score_1 = 0
                    score_2 = 0
                    for re in boundary_nodes:
                        score_1 += self.get_boundary_distance(s1, re)
                        score_2 += self.get_boundary_distance(s2, re)
                    self.add_sup_node(supporting_nodes, candidates,
                                      score_1, score_2, s1, s2, row_names)
                    continue

                n_1 = 0
                n_2 = 0
                di_1 = 0
                di_2 = 0

                searching_ = set([s1])
                searched = set([s1])
                t = 0

                while len(searching_) > 0:
                    t += 1
                    new_searching = set()
                    for node_i in searching_:

                        for node_j in range(np.size(R[node_i])):
                            if R[node_i, node_j] > 0 and node_j not in searched:
                                n_1 += 1
                                di_1 += np.linalg.norm(
                                    data.values[node_i] - data.values[node_j]) / t
                                searched.add(node_j)
                                new_searching.add(node_j)
                    searching_ = new_searching

                searching_ = set([s2])
                searched = set([s2])
                t = 0
                while len(searching_) > 0:
                    t += 1
                    new_searching = set()
                    for node_i in searching_:
                        for node_j in range(np.size(R[node_i])):
                            if R[node_i, node_j] > 0 and node_j not in searched:
                                n_2 += 1
                                di_2 += np.linalg.norm(
                                    data.values[node_i] - data.values[node_j]) / t
                                searched.add(node_j)
                                new_searching.add(node_j)
                    searching_ = new_searching

                centrality_1 = di_1 / n_1
                centrality_2 = di_2 / n_2

                score_1 = (ave_neighbor_degree_1 / (ave_neighbor_degree_1 + ave_neighbor_degree_2) +
                           centrality_2 / (centrality_1 + centrality_2)) / 2
                score_2 = 1 - score_1
                # print('look 397', score_1, score_2)
                self.s_score[s1] = score_1
                self.s_score[s2] = score_2
                self.add_sup_node(supporting_nodes, candidates,
                                  score_1, score_2, s1, s2, row_names)
        # self.cont += 1
        # print(self.cont, supporting_nodes)
        return supporting_nodes

    def get_supporting_nodes_sparse(self, data, R, row_names, boundary_nodes):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        # print(R)
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                s2 = R[s1].argmax()  # s1和s2 是一对RNNs
                degree_1 = R[s1].sum()
                degree_2 = R[s2].sum()
                # self.add_sup_node(supporting_nodes, candidates, random.random(), random.random(), s1, s2, row_names)
                # continue
                # print('look')
                if degree_1 == 2 and degree_2 == 2:  # 如果是孤立的一堆RNNs，直接判断点对位置，忽略后续计算。
                    score_1 = 0
                    score_2 = 0
                    for re in boundary_nodes:
                        score_1 += self.get_boundary_distance(s1, re)
                        score_2 += self.get_boundary_distance(s2, re)
                    self.add_sup_node(supporting_nodes, candidates,
                                      score_1, score_2, s1, s2, row_names)
                    continue

                n_1 = 0
                n_2 = 0
                di_1 = 0
                di_2 = 0
                for i in range(R[s1].shape[1]):
                    if R[s1, i] > 0:
                        n_1 += 1
                        di_1 += R[i].sum()
                for i in range(R[s2].shape[1]):
                    if R[s2, i] > 0:
                        n_2 += 1
                        di_2 += R[i].sum()

                ave_neighbor_degree_1 = di_1 / n_1
                ave_neighbor_degree_2 = di_2 / n_2

                if ave_neighbor_degree_1 == ave_neighbor_degree_2:
                    score_1 = 0
                    score_2 = 0
                    for re in boundary_nodes:
                        score_1 += self.get_boundary_distance(s1, re)
                        score_2 += self.get_boundary_distance(s2, re)
                    self.add_sup_node(supporting_nodes, candidates,
                                      score_1, score_2, s1, s2, row_names)
                    continue

                n_1 = 0
                n_2 = 0
                di_1 = 0
                di_2 = 0

                searching_ = set([s1])
                searched = set([s1])
                t = 0

                while len(searching_) > 0:
                    t += 1
                    new_searching = set()
                    for node_i in searching_:

                        for node_j in range(R[node_i].shape[1]):
                            if R[node_i, node_j] > 0 and node_j not in searched:
                                n_1 += 1
                                di_1 += np.linalg.norm(
                                    data.values[node_i] - data.values[node_j]) / t
                                searched.add(node_j)
                                new_searching.add(node_j)
                    searching_ = new_searching

                searching_ = set([s2])
                searched = set([s2])
                t = 0
                while len(searching_) > 0:
                    t += 1
                    new_searching = set()
                    for node_i in searching_:
                        for node_j in range(R[node_i].shape[1]):
                            if R[node_i, node_j] > 0 and node_j not in searched:
                                n_2 += 1
                                di_2 += np.linalg.norm(
                                    data.values[node_i] - data.values[node_j]) / t
                                searched.add(node_j)
                                new_searching.add(node_j)
                    searching_ = new_searching

                centrality_1 = di_1 / n_1
                centrality_2 = di_2 / n_2

                score_1 = (ave_neighbor_degree_1 / (ave_neighbor_degree_1 + ave_neighbor_degree_2) +
                           centrality_2 / (centrality_1 + centrality_2)) / 2
                score_2 = 1 - score_1
                # print('look 397', score_1, score_2)
                self.s_score[s1] = score_1
                self.s_score[s2] = score_2
                self.add_sup_node(supporting_nodes, candidates,
                                  score_1, score_2, s1, s2, row_names)
        # self.cont += 1
        # print(self.cont, supporting_nodes)
        return supporting_nodes


def get_tree(edges):
    clustering_tree = {}
    roots = set()
    for c, p in edges.items():
        if c == p:
            # print(c, ':', p)
            roots.add(p)

        if p not in clustering_tree.keys():
            nc = set()
            nc.add(c)
            clustering_tree[p] = nc
        else:
            clustering_tree[p].add(c)
    # print(clustering_tree)
    return clustering_tree, roots


def disturb_data(data: pd.DataFrame):
    d = data.values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            d[i, j] += 0.00001 * random.random()
    # get the name of rows
    return d, data.index.values.tolist()


def get_adjacent_matrix_sparse(d):
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors.fit(d)
    # A = neighbors.kneighbors_graph(d) - scipy.sparse
    A = neighbors.kneighbors_graph(
        d) - scipy.sparse.eye(d.shape[0], d.shape[0])
    R = A + A.T
    return A.tocsr(), R.tocsr()


def get_adjacent_matrix(d):
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors.fit(d)
    A = neighbors.kneighbors_graph(d) - np.eye(len(d))
    # A = neighbors.kneighbors_graph(d) - scipy.sparse.eye(len(d))
    R = A + A.T
    return A, R


# def adjacent_list_2_children_map(results):
#     re_map = {}
#     for p in results:
#         re_map[p[0]] = p[1]
#     print(len(results), ',', len(re_map))


def get_labels(clusters, data_size):
    labels = -1 * np.ones(data_size)
    for i, root in zip(range(len(clusters.keys())), clusters.keys()):
        for node in clusters[root].keys():
            labels[node] = i
    return labels


def get_roots_from_edges(edges):
    roots = set()
    for c, p in edges.items():
        if c == p:
            roots.add(p)
    return roots


def edges2community(edges):
    comm = {}
    roots = get_roots_from_edges(edges)
    for r in roots:
        comm[r] = set([r])
    trees = edges2tree(edges)
    for comm_root, comm_nodes in comm.items():
        candidates = trees[comm_root]
        while len(candidates) > 0:
            comm_nodes.update(candidates)
            next_candidates = set()
            for node in candidates:
                if node in trees.keys():
                    next_candidates.update(trees[node])
            candidates = next_candidates - candidates
    return comm


def edges2tree(edges):
    tree = {}
    for c, p in edges.items():
        if p not in tree.keys():
            tree[p] = set()
        tree[p].add(c)
    return tree


# def draw_matrix(m):
#     data = np.array(m)
#     # print(data[:,0])
#     plt.scatter(data[:, 0], data[:, 1])
#     plt.show()

def get_clusters_predict_labels(Graph: nx.Graph):
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    predict_labels = np.zeros(len(Graph.nodes), dtype=int)

    for i, s in enumerate(S[1:], 1):
        predict_labels[list(s.nodes)] = i
    return predict_labels


@retry(retries=3, delay=0)
def run_PRSC(data, theta, loop=1, divide_method='default', K=2):
    _size = math.floor(data.shape[0]*theta)
    num_thread = math.ceil(min(data.shape[0], _size))
    K = num_thread
    for i in range(loop):
        prs = PRS(data)
        prs.get_clusters(num_thread, threshold_clusters=K,
                         divide_method=divide_method)

        prs.get_results()
        ET, roots = prs.get_final_tree_nx()

        # assert len(roots) == K
        # assert len(ET.nodes) == data.shape[0]
        # assert len(ET.edges) == data.shape[0] - K
    # print(f'finish PRSC with divide methodthe {
    #       divide_method} and theta {theta}')
    return ET


if __name__ == '__main__':
    raise Exception('this is a module file, please do not run it directly!')
    # {"breast-w", "ecoli", "glass", "ionosphere", "iris", "kdd_synthetic_control", "mfeat-fourier", "mfeat-karhunen","mfeat-zernike"};
    # {"optdigits", "segment", "sonar", "vehicle", "waveform-5000", "letter", "kdd_synthetic_control"};
    data_names = ["iris", "sonar", "glass", "ecoli", "ionosphere", "kdd_synthetic_control", "vehicle",
                  "mfeat-fourier", "mfeat-karhunen", "mfeat-zernike", "segment", "waveform-5000", "optdigits", "letter", "avila"]
    data_Ks = [3, 2, 2, 8, 2, 6, 4, 10, 10, 10, 7, 3, 9, 26, 12]
    evaluation = {'Name': [], 'ARI': [],
                  'NMI': [], 'RI': [], 'K': [], 'Theta': []}
    os.makedirs('result/PRSC', exist_ok=True)
    datapath = 'G:/data/datasets/UCI/middle/data/'
    file_names = os.listdir(datapath)
    info = {'Dataset': [], 'Samples': [], 'Features': [], 'Class': [], }

    for i in range(len(file_names)):
        df = {"ARI": [], "RI": [], "NMI": [], "K": [], "theta": []}
        # data_name = data_names[i]
        data_name = file_names[i].split('.')[0]
        K = data_Ks[i]
        # file_name = 'exp_disturbed/' + data_name + '.txt'
        file_name = file_names[i]
        rdata, label, K = DataLoader.get_data_from_local(
            f'{datapath}{file_name}')

        info['Dataset'].append(data_name)
        info['Samples'].append(rdata.shape[0])
        info['Features'].append(rdata.shape[1])
        info['Class'].append(K)

        if len(label) < 1e3:
            continue
        # print(data_name)
        theta = 1
        flag = True
        ts = 1
        RI_ave = 0
        ARI_ave = 0
        NMI_ave = 0

        data = rdata.copy()
        data = (data - data.mean()) / (data.std())
        for t in range(ts):
            start = time.perf_counter()
            ET = run_PRSC(data, label, K, theta)
            end = time.perf_counter()

            # print('cpu time:', end - start)
            # r = np.array(sorted(r.items(), key=lambda item: item[0]))[:, 1]
            # rm = np.array(sorted(rm.items(), key=lambda item: item[0]))[:, 1]
            # print(f'the diff of r and rm is {np.sum(r-rm)}')
            # print(r)
            # print('waite for estimation!')

            # r = r.tolist()
            # print(f'len of r{len(r)} and k {k}')
            # ri = rand_score(label[:len(r)], r)
            # ari = adjusted_rand_score(label[:len(r)], r)
            # nmi = normalized_mutual_info_score(label[:len(r)], r)

            ri = rand_score(label, get_clusters_predict_labels(ET))
            ari = adjusted_rand_score(label, get_clusters_predict_labels(ET))
            nmi = normalized_mutual_info_score(
                label, get_clusters_predict_labels(ET))

            df["ARI"].append(ari)
            df["RI"].append(ri)
            df["NMI"].append(nmi)
            df["K"].append(K)
            df["theta"].append(theta)

            RI_ave = ri + RI_ave
            ARI_ave = ari + ARI_ave
            NMI_average = nmi + NMI_ave
            # print(end - start)
        pd.DataFrame(df).to_csv(f"result/PRSC/{data_name}.csv", index=False)
        if flag == True:
            evaluation['Name'].append(data_name)
            evaluation['ARI'].append(ARI_ave / ts)
            evaluation['NMI'].append(NMI_ave / ts)
            evaluation['RI'].append(RI_ave / ts)
            evaluation['K'].append(K)
            evaluation['Theta'].append(theta*100)

            print(data_name, '\tRI : ', RI_ave / ts)
            print(data_name, '\tARI: ', ARI_ave / ts)
            print(data_name, '\tNMI: ', NMI_ave / ts)

        pd.DataFrame(info).to_csv('DataInfo.csv', index=False)
        # output.close()
        eval = pd.DataFrame(evaluation)
        # evaluation 根据RI排序，降序排列
        eval = eval.sort_values(by='RI', ascending=False)
        eval.to_csv('PRSCevaluation.csv', index=False)
