# Gunicorn configuration file
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 4
worker_class = "sync"  # Use sync worker instead of gevent
timeout = 120