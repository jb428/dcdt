FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /root/miniconda && \
    rm miniconda.sh
ENV PATH="/root/miniconda/bin:$PATH"

COPY env_torch_cuda.yml /tmp/env_torch_cuda.yml
RUN conda env create -f /tmp/env_torch_cuda.yml
SHELL ["conda", "run", "-n", "torch_cuda", "/bin/bash", "-c"]

RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install boto3 psycopg2-binary

RUN cd /tmp && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x mc && \
    mv mc /usr/bin/mc 

#ENTRYPOINT ["conda", "run", "-n", "torch_cuda", "--no-capture-output", "/bin/bash"]
CMD ["conda", "run", "-n", "torch_cuda", "/bin/bash"]