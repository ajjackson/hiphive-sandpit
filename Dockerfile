FROM mambaorg/micromamba
COPY env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml &&\
    micromamba clean --all --yes

USER root
RUN apt-get update && apt-get install -y \
    ssh librdmacm1  # Packages needed by openmpi/asap3

# Enable conda environment before using pip for remaining packages
USER mambauser
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install calorine tqdm

COPY --chown=$MAMBA_USER:$MAMBA_USER emt_hiphive_tests.py /tmp/emt_hiphive_tests.py
