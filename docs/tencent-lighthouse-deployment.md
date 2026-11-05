# Tencent Lighthouse deployment and phone test

This runs the cooking coach and Caddy using `docker-compose.lighthouse.yml`.
Azure chat and vision remain remote API calls. No local inference models or
camera hardware services are needed. This guide is prepared; deployment and
real-phone checks have not been performed.

## Prepare the server

Use a Lighthouse Linux server with Docker Engine 28+ and Docker Compose v2.33.1+
(the compose file uses [`gw_priority`](https://docs.docker.com/reference/compose-file/services/#gw_priority)). Point a domain's DNS A record at the
server's public IPv4 address. Allow inbound TCP 80 and 443 for Caddy; restrict
SSH access to your own IP. The app's port 8000 stays private to Docker.
If another proxy already occupies ports 80/443, integrate this domain there
before starting another Caddy instance.

## Configure and start

### Using the Python desktop app

Run `python app.py` in the local project folder, or double-click
`Start Cooking Coach.pyw` on Windows. Save the key in the **Azure settings** tab.
In **Lighthouse**, fill the public domain, SSH server address, username, SSH port,
optional private-key file and a folder name within the remote user's home.
The same model settings are used for local runs and deployments.

Before the first deployment, connect once with your usual SSH client and verify
the server fingerprint, for example `ssh ubuntu@your-server-ip`. The app uses
existing known hosts and SSH key/agent authentication, with no password prompts.
For an encrypted private key, unlock it in your SSH agent first. The remote user
must be able to use Docker without an interactive sudo password.

Click **Check server**, then **Deploy to Lighthouse**. The app checks Docker
versions, transfers a restricted source archive and `.env.lighthouse` over SSH,
runs Compose with health waiting, and verifies `https://<domain>/health` from
your computer. It preserves Docker data volumes. It replaces the application
files and model settings in the selected folder and recreates the containers.
The app does not provision Lighthouse, install Docker, configure DNS or open ports.

You can also choose **Prepare bundle** without SSH details. It writes the private
`.deployment/lh-cooking.tar.gz` archive, which includes the API key. Copy it to
your prepared Linux server, extract it in a dedicated application folder and run
`sh deploy.sh`. Server connection details are remembered in the gitignored
`.deployment/server.json`; SSH private-key contents are never packaged.

There is no live server yet, so the SSH deployment flow has only been tested with
simulated responses. Complete the phone acceptance checks after your first deployment.

### Using the command line

On the server, in your checkout of this repository:

```bash
cp .env.lighthouse.example .env.lighthouse
nano .env.lighthouse
```

Fill these fields locally on the server; never commit the real settings file:

```dotenv
DOMAIN=cook.your-domain.example
OPENAI_API_KEY=your_actual_hkust_key
AZURE_OPENAI_ENDPOINT=https://hkust.azure-api.net
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_DEPLOYMENT_NAME=gpt-5-mini
VLM_PROVIDER=azure
VLM_MODEL=gpt-5-mini
```

Use deployment names permitted for your HKUST account. The vision deployment
must accept images. Caddy obtains TLS certificates for `DOMAIN` automatically
once DNS and inbound ports work.

```bash
chmod 600 .env.lighthouse
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml config --quiet
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml up -d --build
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml ps
curl -fsS https://cook.your-domain.example/health
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml logs --tail=100 chatbot caddy
```

Avoid sharing unredacted compose output or logs containing conversations.
`/health` reports configuration and app readiness; it does not test Azure access.
An actual chat question and camera question below verify the models.

## Real-phone acceptance checks

- [ ] Open the HTTPS domain in the phone's browser; no certificate warning.
- [ ] Grant camera/microphone access; see the rear-camera preview.
- [ ] Answer name, experience, allergies and dish; choose garlic pasta.
- [ ] Say "start", "next", "back", "repeat" and "step 4"; confirm the displayed step.
- [ ] Ask a recipe question; text arrives progressively and sentence speech
      starts before the whole answer finishes. It should not repeat at the end.
- [ ] Enable hands-free mode; listening resumes after the final spoken sentence,
      without transcribing the coach's own reply.
- [ ] Ask "what food can you see?" with the camera on; confirm an accurate
      camera observation appears. A preview alone does not verify vision.
- [ ] Start a step timer and check its chime and reset behavior.
- [ ] Reload the page; confirm the recipe and step resume on the same phone.
- [ ] Turn off the network during a reply; see interruption feedback, reconnect,
      then ask again. A partial reply must not advance the step twice.
- [ ] Turn off voice output and camera; typed questions still work.

Record phone model, OS, browser version and any failed checks. Browser speech
recognition and autoplay vary across devices; use typed input where unavailable.

## Updates and data

Conversation profiles and text observations persist in the `lh_cooking_chatbot_data`
Docker volume; Caddy certificates persist in its own volumes. Camera frames are
not saved. Back up the data volume before upgrades. From the server checkout:

```bash
git pull
docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml up -d --build
```

Rebuilding refreshes recipe knowledge. Do not add `down -v` to routine updates;
that deletes persistent volumes. Changes to production env values need `up -d`
to recreate the container, rather than a simple restart.
