try:
    from .needle_utils import *
except ImportError:
    # needle_utils pulls in matplotlib, which is absent from the llamaqkv env and is
    # only needed for NIAH plotting. The DRM/SlimInfer path lives in cache_utils and
    # processor, so a missing plotting stack must not block `import utils.cache_utils`.
    pass
