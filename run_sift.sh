#!/bin/sh

python run.py --dataset sift-128-euclidean --algorithm nsg --force --runs 1
python run.py --dataset sift-128-euclidean --algorithm diskann --force --runs 1
python run.py --dataset sift-128-euclidean --algorithm hnsw --force --runs 1
python run.py --dataset sift-128-euclidean --algorithm faiss-ivf --force --runs 1
python run.py --dataset sift-128-euclidean --algorithm faiss-lsh --force --runs 1
