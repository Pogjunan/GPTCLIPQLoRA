def log_kv(step, **kwargs):
    items = [f"step={step}"]
    items += [f"{k}={v}" for k,v in kwargs.items()]
    print(" | ".join(items))
