⚡ Quickstart

Assuming you already have a project folder on your computer with data, notebooks, or scripts:

### TensorFlow Geo-AI

```
docker compose --profile tf up -d --build
docker exec -it geoai-tf bash
```

### PyTorch Geo-AI

```
docker compose --profile torch up -d --build
docker exec -it geoai-torch bash
```

Your project folder is mounted automatically at /data inside the container.

Any changes or outputs inside /data persist on your host system.

Stop all containers when done:

```
docker compose down
```
