import numpy
import numpy as np
from pynsg import NSG, Metric, create_graph_file
import faiss
import os

from ..base.module import BaseANN


class NSGLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": Metric.L2, "euclidean": Metric.L2}[metric]
        self.L = method_param['L']
        self.R = method_param['R']
        self.C = method_param['C']
        self.knn = method_param['knn']
        self.normalize = metric == "angular"
        self.knn_file = f"knn{self.knn}.graph"


    def fit(self, X: numpy.array):
        if self.normalize:
            faiss.normalize_L2(X)
        # will just build knn graph with metric L2 always since the vectors will be normalized for angular metrics
        create_graph_file(filename=self.knn_file, x=X, k=200)
        self.index = NSG(dimension=len(X[0]), num_points=len(X), metric=self.metric)
        self.index.build_index(data=X, knng_path=f"knn{self.knn}.graph", L=self.L, R=self.R, C=self.C)
        self.index.optimize_graph(data=X)

    def set_query_arguments(self, search_L):
        self.search_L = search_L
        self.name = f"nsg (L={self.L}, R={self.R}, C={self.C}, knn={self.knn}, search_L={search_L})" 

    def query(self, q: numpy.array, n: int) -> numpy.array:
        return self.index.search_opt(queries=np.expand_dims(q, axis=0), k=n, search_L=self.search_L)[0]

    def done(self):
        if os.path.exists(self.knn_file):
            os.remove(self.knn_file)

