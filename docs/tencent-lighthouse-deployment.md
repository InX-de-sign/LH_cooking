# Tencent Lighthouse Deployment

This deployment keeps Azure OpenAI as the chatbot model provider and uses a
Tencent Lighthouse instance in Hong Kong to run the application containers.

## Before deployment

1. Create an Ubuntu 22.04 or 24.04 Lighthouse instance with at least 4 vCPU,
   8 GB RAM, and 60 GB SSD. CPU inference is supported; no GPU is required.
2. Point an `A` record for the chosen domain to the Lighthouse public IPv4
   address. DNS must resolve before Caddy can obtain its TLS certificate.
3. In the Lighthouse firewall, allow inbound TCP `22`, `80`, and `443`.
   Keep `5000`, `5432`, `6379`, and `6333` closed. The production stack does
   not expose those ports.
4. Rotate any API keys that were committed or copied into older `.env` files.
   Never commit `.env.lighthouse`.

## Install Docker on Lighthouse

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker
docker compose version
```

## Deploy

```bash
git clone <your-private-repository-url> pal-op
cd pal-op
cp .env.lighthouse.example .env.lighthouse
nano .env.lighthouse
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml up -d --build
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml ps
```

The first build can take several minutes because it installs PyTorch,
Ultralytics, and OpenCV. The model file `inference/models/best_animal_v3.onnx`
must be present in the repository or copied to that path before startup.

## Verify

```bash
curl -fsS https://your-domain.example/health
curl -fsS https://your-domain.example/vision/health
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml logs --tail=100 chatbot yolo_inference caddy
```

The web interface and chatbot WebSockets use the domain root. ESP32 camera
uploads and WebSockets use the same domain under `/vision/...`, for example:

```text
wss://your-domain.example/vision/ws/esp32/camera/esp32_camera_1
https://your-domain.example/vision/upload_batch
```

Update the ESP32 firmware host from the old Tailscale hostname to the new
domain and retain port `443`. Use TLS-capable client methods in the firmware.

## RTSP input

MediaMTX is private to the Docker network by default. This is appropriate for
ESP32 image uploads. If an external camera must publish RTSP to Lighthouse,
add a `ports` mapping for `8554:8554` to the `mediamtx` service and allow TCP
8554 only from the camera's fixed source IP in both the Lighthouse firewall
and the host firewall.

## Persistent data and updates

`chatbot_data` stores the SQLite conversation database at
`/app/data/memory.db`; `chatbot_logs` stores application logs. Back up both
volumes before upgrades:

```bash
docker run --rm -v pal_op_chatbot_data:/source -v "$PWD":/backup alpine \
  tar czf /backup/chatbot-data-backup.tgz -C /source .
docker run --rm -v pal_op_chatbot_logs:/source -v "$PWD":/backup alpine \
  tar czf /backup/chatbot-logs-backup.tgz -C /source .
```

To update the application:

```bash
git pull
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml up -d --build
```