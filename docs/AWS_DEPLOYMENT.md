# OrderOrder — AWS deployment

| | |
|---|---|
| **Version** | 0.1 |
| **Date** | 11 September 2026 |
| **Companion documents** | [DEPLOYMENT.md](DEPLOYMENT.md) — the constraints this inherits · [TECH_STACK.md](TECH_STACK.md) · [ARCHITECTURE.md](ARCHITECTURE.md) §10 |

This document takes the serving profile — one container, one process, one SQLite file — and puts it on
AWS. It inherits every constraint in [DEPLOYMENT.md](DEPLOYMENT.md) and adds none: the same image, the
same token rule, the same single worker.

---

## 1. What is being deployed

The shape of the thing decides the topology, so it is worth stating plainly:

| | | AWS consequence |
|---|---|---|
| **One process, one worker** | Jobs live in an in-process dictionary (`web/jobs.py`); a second worker means job lookups 404 at random (DEPLOYMENT §2.3) | **One instance, one container.** No autoscaling group, no `--workers`, no ECS service with `desiredCount > 1` |
| **One SQLite file, one writer** | ~3 KB per paragraph; the full 1950–2025 corpus is 707,647 paragraphs, so budget **~2–3 GB and growing** (§2.6) | The corpus is an **EBS volume**, not an image layer and not ephemeral container storage |
| **Published to loopback only** | The compose `serve` profile publishes `127.0.0.1:8000`; TLS is the reverse proxy's decision (§2.2) | Port 8000 is **never in the security group**. TLS terminates at a proxy on the box, or at an ALB |
| **A token or it does not start** | `ORDERORDER_API_TOKEN` has no default; an empty token is no token (§2.2) | The token arrives as environment, from SSM or a root-owned `.env` — never baked, never committed |
| **`/api/health` answers without a token** | Carved out deliberately for a load balancer (§2.2) | An ALB target group can health-check it as-is |
| **The model is external** | The only tier that serves is a **paid API** (§2.1); a free tier degrades to *not checked* | Keys are env vars on the instance. No GPU is needed on AWS unless you self-host the model |
| **Ingestion is anonymous HTTP** | The AWS Open Data bucket needs no AWS account or IAM role | The instance needs **no S3 permissions to ingest**. IAM is only for your own corpus copy and secrets |

The deployment is therefore: **one EC2 instance (or one Lightsail instance), one gp3 EBS volume for
`/data`, Docker running the `serve` compose profile, and a reverse proxy for TLS.** That is not a
compromise; it is what the architecture asks for.

### What not to choose, and why

| Option | Why not |
|---|---|
| **ECS Fargate, multi-task** | Two tasks = two workers = job lookups 404 at random (§2.3). Fargate tasks are also ephemeral; the corpus is 2 GB of state that must live on a volume, and the bind-mount uid-10001 ownership rule (§3) becomes an exercise in EFS permissions for no gain |
| **Lambda** | A 2 GB corpus loaded per invocation, a process that must persist between requests to serve job progress, and a static page that wants one origin. Wrong shape entirely |
| **Autoscaling / multiple AZs** | The rate limiter and the job registry both live in one process (§3a). Horizontal scale changes the design, not the instance count — ARCHITECTURE §4.13 is that change, and it is not built |
| **RDS for the pilot** | Optional later (§6). SQLite reads are concurrent and the writes are ingestion, which you do once. A managed database earns its keep only when a second process wants the corpus |

---

## 2. The target topology

```
                Internet
                   │ 443
            ┌──────┴──────┐
            │  Nginx/Caddy │  on the instance (Let's Encrypt)
            │  or an ALB   │  (ACM certificate)
            └──────┬──────┘
                   │ 127.0.0.1:8000          ┌─────────────────────┐
            ┌──────┴──────────┐              │ LLM provider (paid) │
            │  orderorder     │ ───────────► │ Gemini / Groq / …   │
            │  container      │              └─────────────────────┘
            │  uid 10001      │
            │  one worker     │
            └──────┬──────────┘
                   │ bind mount
          ┌────────┴─────────┐
          │  EBS gp3 volume  │
          │  /srv/orderorder │
          │  -data (2–3 GB+) │
          │  SQLite corpus   │
          └──────────────────┘
```

### Sizing

| | pilot | comfortable |
|---|---|---|
| Instance | `t3.medium` (2 vCPU, 4 GB) | `t3.large` / `m7i.large` (2 vCPU, 8 GB) |
| EBS gp3 | 30 GB | 50 GB (corpus + Postgres if switched + model caches) |
| Region | `ap-south-1` (Mumbai) — the users are Indian lawyers; latency to them is what a region buys | |

The engine is CPU-bound lexical search over an in-process index with an external model; 4 GB is enough
to serve. What wants more CPU is the **one-time corpus build** — 65 judgments a minute on eight
workers, ~10 hours for 38,032 judgments — which §3.2 handles with a throwaway instance rather than a
permanently larger one.

Rough monthly cost, `ap-south-1`, on-demand: `t3.medium` ≈ **$25**, 30 GB gp3 ≈ **$3**, so the pilot
is **under $30 a month** before data transfer. A Lightsail 4 GB instance ($24, includes an 80 GB SSD
and 4 TB transfer) is the same design with less to manage — take it if the console is the interface
you want. The ALB alternative to on-box TLS is ≈ $16 a month more.

---

## 3. Setup, in order

### 3.1 The instance

1. **EC2 → Launch instance.** Amazon Linux 2023, `t3.medium`, `ap-south-1`, key pair of your choice.
2. **Storage:** 30 GB gp3. This is the root volume; the corpus gets its own space on it under
   `/srv/orderorder-data` (a separate EBS volume is fine and slightly cleaner to snapshot, but one
   volume is simpler and enough for the pilot).
3. **Security group** — inbound:

   | Port | Source | For |
   |---|---|---|
   | 22 | your IP only | SSH |
   | 443 | 0.0.0.0/0 | HTTPS |
   | 80 | 0.0.0.0/0 | redirect to 443 |

   **Not 8000.** The container publishes to `127.0.0.1:8000` on purpose (§2.2); a security group that
   allows 8000 is a second, accidental exposure the design already refused.

4. SSH in and install Docker (Amazon Linux 2023):

   ```bash
   sudo dnf install -y docker git
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER && exit   # re-login for the group to apply
   ```

### 3.2 The corpus — build once, on a box sized for it

You have never built the corpus inside the container (DEPLOYMENT §3 says so), and the build is CPU-
and network-bound, not storage-bound. Two ways:

**Option A — build on a throwaway instance, snapshot the result (recommended).**
The build is a ten-hour, one-time job; run it on an instance you delete.

```bash
# Locally sized for the build, then deleted:
#   EC2: t3.xlarge (4 vCPU, 16 GB), Amazon Linux 2023, same AMI
sudo dnf install -y docker git && sudo systemctl enable --now docker

git clone <your-repo-url> order-order && cd order-order
export ORDERORDER_DATA_DIR=/srv/orderorder-data
sudo install -d -o 10001 -g 10001 $ORDERORDER_DATA_DIR

# Ingestion needs Python + uv, not the container — the corpus is built before the image runs:
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
uv run orderorder init-db
uv run orderorder ingest metadata 1950 1951 1952 ...   # or a year loop; ~15 min for all 76 years
uv run orderorder ingest bulk-text --limit 20000        # resumable; repeat, or run without --limit
uv run orderorder ingest aliases
uv run orderorder ingest repair-trailers && uv run orderorder ingest mark-roles
uv run orderorder index && uv run orderorder citator && uv run orderorder embed
uv run orderorder stats                                 # check before serving (DEPLOYMENT §3)
```

Then move the result to the serving instance — an **EBS snapshot** of the data volume is the
mechanism AWS gives you (snapshot the volume built above, create a volume from it, attach to the
serving instance at the same path), or `rsync -a` the directory over SSH, or upload the SQLite file
to a **private S3 bucket** and download it on the box:

```bash
# S3 route — needs an instance role with s3:GetObject on the bucket (§3.4)
aws s3 cp s3://<your-bucket>/orderorder-data.tar.gz - | sudo tar -xzf - -C /srv/
sudo chown -R 10001:10001 /srv/orderorder-data
```

**Option B — build on the serving instance.** Same commands, on the `t3.medium`, over ten hours.
Nothing wrong with it except that it is slow; the run is resumable, so even a flaky box gets there.
Do **not** run the build while the container is serving — SQLite has one writer (§2.4).

A third option exists — copy an existing 1.2 GB SQLite file from a machine that already holds the
corpus — and then, before anything else:

```bash
uv run orderorder migrate --stamp    # a database built before migrations: claim it, apply nothing (§2.5)
```

Which of `migrate` and `migrate --stamp` to run is the whole decision, and DEPLOYMENT §2.5 says it
in full: a database that already holds the corpus gets `--stamp`, an empty one gets plain `migrate`.

### 3.3 The data directory

The image runs as **uid 10001**, not root, and the compose volume is a **bind mount** — so the host
directory's ownership is what the container sees (§3):

```bash
sudo install -d -o 10001 -g 10001 /srv/orderorder-data
```

Skip this and the failure arrives at the first request rather than at boot, which is exactly what
makes it look like a bug in the engine instead of a permissions decision.

### 3.4 Secrets — SSM Parameter Store, not `.env` on disk

Keys never enter an image layer (§3), and AWS gives you a better place than a file: **Systems
Manager Parameter Store** (free, KMS-encrypted by default). Put the token and provider keys there
once:

```bash
aws ssm put-parameter --name /orderorder/api-token  --value $(openssl rand -hex 32) --type SecureString
aws ssm put-parameter --name /orderorder/google-key --value $GOOGLE_API_KEY --type SecureString
# ... groq, cerebras as needed
```

Give the instance an **IAM role** (not access keys) whose policy allows `ssm:GetParameter` on
`/orderorder/*`, then materialise them at boot:

```bash
# /usr/local/bin/orderorder-env — fetches secrets into the environment compose interpolates from
#!/bin/bash
TOKEN=$(aws ssm get-parameter --name /orderorder/api-token --with-decryption --query Parameter.Value --output text)
GOOGLE=$(aws ssm get-parameter --name /orderorder/google-key --with-decryption --query Parameter.Value --output text)
export ORDERORDER_API_TOKEN="$TOKEN" GOOGLE_API_KEY="$GOOGLE" LLM_PRIMARY=google_genai:gemini-3.6-flash
exec docker compose -f /opt/order-order/infra/docker-compose.yml --profile serve up -d
```

If this is more ceremony than you want, a `.env` beside the compose file at mode `600`, root-owned,
is acceptable for a pilot — compose reads it for interpolation. What is not acceptable, in either
route, is the token being empty: compose substitutes an empty string for an unset variable and
`serve` refuses to boot, by design.

**Probe the model before trusting it** (§2.1 — the costliest failure this project has met was a
retired model id reading as an empty corpus):

```bash
docker exec <container> orderorder doctor --probe
```

### 3.5 The container

```bash
git clone <your-repo-url> /opt/order-order && cd /opt/order-order
export ORDERORDER_DATA_DIR=/srv/orderorder-data
export ORDERORDER_API_TOKEN=$(aws ssm get-parameter --name /orderorder/api-token \
    --with-decryption --query Parameter.Value --output text)

docker compose -f infra/docker-compose.yml --profile serve up -d --build
curl -s localhost:8000/api/health          # no token needed — this is the carved-out route
curl -s -H "Authorization: Bearer $ORDERORDER_API_TOKEN" \
     "localhost:8000/api/search?q=a+misrepresentation+vitiates+consent"
```

The compose file already pins everything AWS-relevant: port `127.0.0.1:8000`, `/data` bind mount from
`ORDERORDER_DATA_DIR`, token with no default, `restart: unless-stopped`. If you switched the database
to Postgres (§6), `DATABASE_URL` is the one variable to add.

**One worker.** The compose profile never passes `--workers` and must stay that way (§2.3). If you
ever run uvicorn directly, that is the trap.

### 3.6 TLS — two routes

**Caddy on the box (simplest).** One file, automatic Let's Encrypt:

```bash
sudo dnf install -y caddy
```

```
# /etc/caddy/Caddyfile
orderorder.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

Point the DNS A record at the instance's public IP (or an Elastic IP — reserve one, or a stop/start
changes it and the DNS goes stale), `sudo systemctl reload caddy`. HSTS belongs here, not in the
app (§3a, item 19).

**ALB + ACM (when you want AWS to hold the certificate).** Create an ACM certificate for the domain
(DNS-validated), an internet-facing ALB with an HTTPS:443 listener on it, and a target group of type
`instance` on port **8000**… but note that means the ALB talks to port 8000 directly, so either
change the compose publish to bind the ALB's security group only, or keep the loopback publish and
put the ALB's SG as the sole source of port 8000 on the instance SG. Health check: `GET /api/health`,
expected 200 — it is the one route that answers without a token, which is why it exists.

---

## 4. Operations on AWS

| | |
|---|---|
| **Logs** | `docker logs` — one stderr logger, `ORDERORDER_LOG_LEVEL` for volume, prompts never logged (§2.3a). Send to CloudWatch with the agent if retention matters |
| **Health** | `/api/health` from outside via the proxy, or the Docker `HEALTHCHECK` already in the image |
| **The number to watch** | **The abstention rate.** A throttled or dead model produces honest *not checked* verdicts that read like data (§2.1, §4). Probe after deploy, and re-probe when a report comes back emptier than the last one |
| **Backups** | The corpus is one directory. **EBS snapshots** on a schedule (DLifecycle Manager, weekly is plenty — the corpus changes only when you ingest) are the whole answer |
| **Updates** | `git pull && docker compose -f infra/docker-compose.yml --profile serve up -d --build`. Migrations run per §2.5 |
| **Spot/hibernate** | Fine for the build box. Do not put the serving instance on spot — a job in flight dies with the instance, and jobs are the product |

## 5. Cost summary

| Item | Monthly (ap-south-1, on-demand) |
|---|---|
| `t3.medium` serving instance | ~$25 |
| 30 GB gp3 | ~$3 |
| ALB (only if chosen over Caddy) | ~$16 |
| SSM Parameter Store (standard tier) | $0 |
| **One-time:** `t3.xlarge` build box for ~2 days | ~$7, then deleted |
| LLM provider | Per DEPLOYMENT §2.1 — a paid tier is the only one that serves; this is the real bill |

Lightsail 4 GB replaces the first two rows with $24 and less to operate.

## 6. Optional: RDS Postgres instead of SQLite

`infra/docker-compose.yml` already carries `pgvector/pgvector:pg17` in the `hackathon`/`prod`
profiles and `DATABASE_URL` switches to it. On AWS the same move is an **RDS PostgreSQL 17 instance
with the pgvector extension** — but DEPLOYMENT §2.4 is the constraint that survives the move: **the
corpus must be re-ingested rather than copied**, so it is a ten-hour rebuild against the RDS
endpoint, not a `pg_dump`. Take this path only when a second process needs the corpus (ARCHITECTURE
§4.13), which is the same day the single-worker rule goes. Until then it is a second database to
pay for and nothing to query it.

## 7. Before anyone else uses it

Inherited from [DEPLOYMENT.md](DEPLOYMENT.md) §4, none of it optional on a public box:

- **Attribution.** CC-BY-4.0 for the judgment data — the README attribution belongs in the footer of
  anything that serves the corpus.
- **The three states stay three states.** Supported / checked-and-not-supported / *not checked*. A
  deployment that flattens the third is the failure the engine exists to prevent.
- **Uploaded briefs are privileged.** They are never stored, and `LLM_SENSITIVE` names the only
  provider allowed to see non-demo text. On AWS this also means: CloudWatch logs are safe (prompts
  are never logged, `tests/test_logs.py` holds it), but do not add request-body logging at the proxy
  to "debug" it.
- **`docker history` on anything you push.** If the image ever goes to ECR, the no-keys-in-layers
  rule (§3a item 1) applies to the registry too.
