FROM ann-benchmarks

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libmkl-full-dev
RUN pip install numpy faiss-cpu

RUN python -c 'import faiss; print(faiss.IndexFlatL2); import numpy as np; np.__config__.show()'
