# geoai-tf-gpu
FROM tensorflow/tensorflow:2.13.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# System dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Geospatial Python packages
RUN pip install --no-cache-dir \
    geopandas fiona==1.9.6 \
    rioxarray s2sphere \
    dask xarray rasterstats \
    pystac-client planetary-computer stackstac

# ML/AI packages
RUN pip install --no-cache-dir \
    pandas scipy scikit-learn \
    matplotlib tqdm \
    segmentation-models

WORKDIR /workspace
CMD ["/bin/bash"]
