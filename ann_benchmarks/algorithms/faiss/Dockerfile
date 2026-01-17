FROM ann-benchmarks

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libmkl-full-dev
RUN pip install faiss-cpu

RUN python -c 'import faiss; print(faiss.IndexFlatL2); np.__config__.show()'
