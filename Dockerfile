FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --only-binary=:all: --no-binary=rouge-score \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.2.2+cpu" -r requirements.txt \
 && python -c "import torch, numpy, pandas, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
 && find /usr/local/lib/python3.10 -name "__pycache__" -prune -exec rm -rf {} + \
 && rm -rf /usr/local/lib/python3.10/site-packages/torch/test \
           /usr/local/lib/python3.10/site-packages/torch/include

COPY fedrag_sim.py fl_net.py fqp.py textgen.py textnet.py hypo_learn.py measurements.py sparql_ext.py edge_controller.py connection_module.py cache_store.py ./
RUN mkdir -p /work /results
ENTRYPOINT ["python", "-u", "fl_net.py"]