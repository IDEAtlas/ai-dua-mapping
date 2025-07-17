#### installation to build the docker image

docker build -f Dockerfile.torch-geoai -t torch-geoai .
docker build -f Dockerfile.tf-geoai -t tf-geoai .
docker build -f Dockerfile.terrakit -t terrakit .

#### instructions to run the docker image

docker run -d --name torch-geoai --gpus all -v /data:/data torch-geoai tail -f /dev/null
docker run -d --name tf-geoai --gpus all --shm-size=8g -v /data:/data geomercato/pt-geoai:1.0 tail -f /dev/null
docker run -d --name pt-geoai --gpus all --shm-size=8g -v /data:/data geomercato/pt-geoai:1.0 tail -f /dev/null



#### instructions to interact with the docker container

docker exec -it torch-geoai bash
docker exec -it tf-geoai bash
docker exec -it terrakit bash