FROM mcr.microsoft.com/devcontainers/miniconda:1-3

COPY environment.yml /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
      /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; \
    fi \
    && rm -rf /tmp/conda-tmp
