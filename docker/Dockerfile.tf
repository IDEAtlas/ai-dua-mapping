# geoai-tf-gpu
FROM tensorflow/tensorflow:2.13.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# System dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev \
    proj-bin libproj-dev \
    libspatialindex-dev \
    libgl1 libglib2.0-0 \
    curl git wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Geospatial Python packages
RUN pip install --no-cache-dir \
    geopandas==0.13.2 \
    rioxarray \
    dask xarray rich \
    pystac-client planetary-computer stackstac \
    osmnx momepy

# Common ML/AI packages
RUN pip install --no-cache-dir \
    pandas scipy scikit-learn seaborn statsmodels \
    matplotlib tqdm pillow opencv-python-headless \
    segmentation-models

WORKDIR /data
CMD ["/bin/bash"]
