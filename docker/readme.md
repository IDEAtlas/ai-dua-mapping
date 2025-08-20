#### Build the docker image

- docker build -f Dockerfile.torch-geoai -t torch-geoai .
- docker build -f Dockerfile.tf-geoai -t tf-geoai .
- docker build -f Dockerfile.terrakit -t terrakit .

#### Run the docker image

- docker run -d --name torch-geoai --gpus all -v /data:/data torch-geoai tail -f /dev/null
- docker run -d --name tf-geoai --gpus all -v /data:/data tf-geoai tail -f /dev/null
- docker run -d --name terrakit  -v /data:/data -v /eodata:/eodata terrakit tail -f /dev/null

#### Interact with the docker container

- docker exec -it torch-geoai bash
- docker exec -it tf-geoai bash
- docker exec -it terrakit bash

