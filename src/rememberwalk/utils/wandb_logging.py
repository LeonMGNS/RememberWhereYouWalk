"""Helpers for logging a training run to wandb (or not at all)."""

# Config values wandb can store; everything else is dropped by filter_nested
allowed_types = (int, float, str, bool)


def filter_nested(obj):
    # Keep simple values
    if isinstance(obj, allowed_types):
        return obj

    # Recurse into dictionaries
    if isinstance(obj, dict):
        return {
            k: v_filtered
            for k, v in obj.items()
            if (v_filtered := filter_nested(v)) is not None
        }

    # Recurse into lists/tuples
    if isinstance(obj, (list, tuple)):
        result = [filter_nested(item) for item in obj]
        return [item for item in result if item is not None]

    # Everything else is removed
    return None


class EmptyLogger:

    def __init__(self):
        pass
        
    def log(self,metrics,step):
        return
