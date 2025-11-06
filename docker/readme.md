⚡ Quickstart

Assuming you already have a project folder on your computer with data, notebooks, or scripts:

### Geo-AI with TensorFlow

```
docker compose --profile tf up -d --build
docker exec -it geoai-tf bash
```


### Terrakit for processing sentinel and other spatial data

```bash
docker compose --profile terrakit up -d --build
docker exec -it terrakit bash
```

Or build all of them at once:
```bash
docker compose --profile tf --profile terrakit up --build -d
```
Your project folder is mounted automatically at /data inside the container.

Any changes or outputs inside /data persist on your host system.

Stop all containers when done:

```
docker compose down
```
