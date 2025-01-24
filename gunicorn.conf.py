workers = 4
bind = "0.0.0.0:$PORT"
timeout = 120  # Longer timeout for LLM responses
worker_class = 'gevent'  # For async support