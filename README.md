# EMNIST Character Recognition

A CNN trained on the EMNIST Balanced dataset, served through a FastAPI endpoint. The included Streamlit app is an optional client for uploading or drawing a character.

## Requirements

- Anaconda or Miniforge
- Python 3.12 (created automatically by `environment.yml`)
- The EMNIST Balanced MATLAB dataset

Create and activate the environment from Anaconda Prompt in the project directory:

```powershell
conda env create -f environment.yml
conda activate emnist-api
```

## Get the data and train

Download the MATLAB-format archive from the [official NIST EMNIST page](https://www.nist.gov/itl/products-and-services/emnist-dataset). Extract `emnist-balanced.mat` into a `matlab` folder in the project:

```text
codigo-IA/
  matlab/
    emnist-balanced.mat
```

Train the model and create the API artifacts:

```powershell
python treino.py
```

This writes `modelo_emnist_balanced.keras` and `label_map.json` beside the scripts. The training dataset and generated model files are excluded from Git; train the model locally and transfer those two artifacts to the server.

## Run the API locally

Set a private API key in Anaconda Prompt or PowerShell, then start the server:

```powershell
$env:API_KEY = "replace-with-a-long-random-secret"
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`; the unauthenticated health check is `http://127.0.0.1:8000/health`.

Send an image with the API key in the `X-API-Key` header:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "X-API-Key: replace-with-a-long-random-secret" \
  -F "file=@character.png"
```

The response contains the predicted character, class index, and confidence. Accepted image formats are those supported by Pillow; uploads are limited to 10 MB.

## Run the Streamlit client

Keep the API running in one terminal. In another, set the same key and the API URL, then run:

```powershell
$env:API_KEY = "replace-with-a-long-random-secret"
$env:API_URL = "http://127.0.0.1:8000"
streamlit run teste.py
```

## Deploy on an EC2 instance

1. Launch an Ubuntu 24.04, x86_64 EC2 instance. For inference, use an instance with at least 4 GiB of memory. Training can be done on your computer; the server only needs `modelo_emnist_balanced.keras` and `label_map.json`.
2. In the EC2 security group, allow SSH from your own IP. For a quick public test, also allow inbound TCP port `8000` from the clients that need access. Restricting the source IP is safer than allowing `0.0.0.0/0`.
3. Install Git and Miniforge on the instance, clone this repository, and create the environment:

   ```bash
   sudo apt-get update
   sudo apt-get install -y git wget
   wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3.sh -b -p "$HOME/miniforge3"
   source "$HOME/miniforge3/etc/profile.d/conda.sh"
   git clone YOUR_GITHUB_REPOSITORY_URL codigo-IA
   cd codigo-IA
   conda env create -f environment.yml
   ```

4. Copy the trained artifacts from your computer into the project directory on EC2:

   ```bash
   scp -i YOUR_KEY.pem modelo_emnist_balanced.keras label_map.json ubuntu@EC2_PUBLIC_IP:~/codigo-IA/
   ```

5. Store an API key outside the repository and create a `systemd` service so the API restarts after a reboot:

   ```bash
   sudo tee /etc/emnist-api.env >/dev/null <<EOF
   API_KEY=$(openssl rand -hex 32)
   EOF
   sudo chmod 600 /etc/emnist-api.env
   sudo tee /etc/systemd/system/emnist-api.service >/dev/null <<'EOF'
   [Unit]
   Description=EMNIST Character Recognition API
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/codigo-IA
   EnvironmentFile=/etc/emnist-api.env
   ExecStart=/home/ubuntu/miniforge3/envs/emnist-api/bin/uvicorn api:app --host 0.0.0.0 --port 8000
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   EOF
   sudo systemctl daemon-reload
   sudo systemctl enable --now emnist-api
   sudo systemctl status emnist-api
   ```

The public endpoint is `http://EC2_PUBLIC_IP:8000/predict`. Read the generated key from `/etc/emnist-api.env` and send it in the `X-API-Key` header. Check logs with `sudo journalctl -u emnist-api -f`.

## Security note

Direct public HTTP is suitable only for a temporary demo: the API key is not encrypted in transit. For a real public deployment, put the API behind HTTPS using a domain and a reverse proxy such as Nginx, and expose port 443 instead of exposing Uvicorn directly. Do not commit API keys, dataset files, or trained model artifacts to GitHub.