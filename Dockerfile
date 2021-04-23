#FROM nvcr.io/nvidia/pytorch:21.03-py3
FROM nvcr.io/nvidia/pytorch:21.03-py3
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
