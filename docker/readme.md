#### Build the docker image

- docker build -f Dockerfile.torch-geoai -t torch-geoai .
- docker build -f Dockerfile.tf-geoai -t tf-geoai .
- docker build -f Dockerfile.terrakit -t terrakit .

#### Run the docker image

<<<<<<< HEAD
docker run -d --name torch-geoai --gpus all -v /data:/data torch-geoai tail -f /dev/null
docker run -d --name tf-geoai --gpus all --shm-size=8g -v /data:/data geomercato/pt-geoai:1.0 tail -f /dev/null
docker run -d --name pt-geoai --gpus all --shm-size=8g -v /data:/data geomercato/pt-geoai:1.0 tail -f /dev/null


=======
- docker run -d --name torch-geoai --gpus all -v /data:/data torch-geoai tail -f /dev/null
- docker run -d --name tf-geoai --gpus all -v /data:/data tf-geoai tail -f /dev/null
- docker run -d --name terrakit  -v /data:/data -v /eodata:/eodata terrakit tail -f /dev/null
>>>>>>> c487c72b8fa184c57f75a450e841366edd0421ab

#### Interact with the docker container

- docker exec -it torch-geoai bash
- docker exec -it tf-geoai bash
- docker exec -it terrakit bash
