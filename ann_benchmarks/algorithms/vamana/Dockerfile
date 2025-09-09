FROM ann-benchmarks

RUN apt-get update
RUN apg-get install -y make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
RUN apt-get install libmkl-full-dev

RUN git clone https://github.com/microsoft/diskann
RUN cd diskann && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j

RUN cd diskann/python && pip install -e .
RUN python -c 'import diskannpy'