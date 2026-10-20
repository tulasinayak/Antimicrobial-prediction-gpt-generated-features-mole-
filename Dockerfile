# Reproducible environment for the antimicrobial-prediction experiments.
#
# Every dependency is pinned to the version that produced the reported numbers
# (see requirements-lock.txt). rdkit ships official PyPI wheels, so no conda
# layer is needed and the image stays small.
#
# Build:
#   docker build -t gpt-mole:1.0 .
#
# Reproduce Table 1 (writes to the mounted results directory):
#   docker run --rm \
#       -v "$PWD/results:/work/results" \
#       gpt-mole:1.0 \
#       python -u scripts/run_table1.py --seeds 0,1,2,3,4
#
# The data the pipeline reads is baked into the image from data/, which is
# tracked in git; nothing is downloaded at run time.

FROM python:3.13.5-slim

# OpenMP is required by xgboost; libxrender/libxext by rdkit's drawing code.
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      libgomp1 \
      libxrender1 \
      libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Dependencies first so edits to the code do not invalidate the install layer.
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir --require-hashes=false -r requirements-lock.txt

COPY notebooks/ ./notebooks/
COPY scripts/ ./scripts/
COPY data/ ./data/

# XGBoost is the bottleneck and is already threaded; leave the BLAS libraries
# single-threaded so they do not oversubscribe the same cores.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

# Run as a non-root user; results land in a mounted volume.
RUN useradd --create-home --uid 1000 runner \
 && mkdir -p /work/results \
 && chown -R runner:runner /work
USER runner

CMD ["python", "-u", "scripts/run_table1.py", "--dry-run"]
