# gunicorn_config.py
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
backlog = 2048

# Worker processes
workers = min(multiprocessing.cpu_count() + 1, 4)
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Timeout settings
timeout = 120  # 2 minutes for scale calculations
graceful_timeout = 30
keepalive = 5

# Restart workers periodically
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = 'scale-api'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# Worker temp directory
worker_tmp_dir = "/dev/shm"

# Preload app for better performance
preload_app = True
