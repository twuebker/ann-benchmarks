FROM ann-benchmarks

RUN apt update && apt install -y wget

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh && \
    bash Anaconda3-2020.11-Linux-x86_64.sh -b && \
    rm Anaconda3-2020.11-Linux-x86_64.sh

ENV PATH=/root/anaconda3/bin:$PATH

RUN python -m pip install --no-cache-dir ansicolors==1.1.8 docker==5.0.2

RUN conda install -y -c pytorch faiss-cpu h5py numpy && \
    conda clean -afy

RUN python -c 'import faiss; import numpy as np; print("FAISS OK:", faiss.IndexFlatL2(128)); np.__config__.show()'




