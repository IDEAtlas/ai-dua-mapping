FROM tensorflow/tensorflow:2.13.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    geopandas==0.12.2 \
    fiona==1.9.4 \
    rasterio==1.3.9 \
    rioxarray==0.13.4 \
    s2sphere==0.2.5 \
    dask==2023.3.0 \
    xarray==2023.1.0 \
    rasterstats==0.19.0 \
    pystac-client==0.6.1 \
    odc-stac==0.3.9 \
    odc-geo==0.4.1 \
    scipy==1.10.1 \
    scikit-learn==1.2.2 \
    tqdm==4.65.0 \
    jupyterlab==3.6.6 \
    pyproj==3.4.1 \
    shapely==2.0.1 \
    numpy==1.24.3 \
    segmentation-models==1.0.1 \
    pandas==1.5.3

WORKDIR /ai-dua-mapping
CMD ["/bin/bash"]