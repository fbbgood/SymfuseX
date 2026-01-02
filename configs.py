# -*- coding: utf-8 -*-
from yacs.config import CfgNode as CN

_C = CN()

# -------------------------
# Drug feature extractor
# -------------------------
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75
_C.DRUG.PADDING = True
_C.DRUG.HIDDEN_LAYERS = [128, 128, 128, 128, 128]
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.MAX_NODES = 290

# -------------------------
# Protein feature extractor
# -------------------------
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True

# -------------------------
# SYMBIOSIS / XFUSION
# -------------------------
_C.SYMBIOSIS = CN()
_C.SYMBIOSIS.HIDDEN_DIM = 256
_C.SYMBIOSIS.NUM_HEADS  = 4
_C.SYMBIOSIS.NUM_LAYERS = 2
_C.SYMBIOSIS.DROPOUT    = 0.10

# -------------------------
# Decoder
# -------------------------
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 1

# -------------------------
# Preprocess (new)
# -------------------------
_C.PREPROCESS = CN()
_C.PREPROCESS.ENABLE = True    
_C.PREPROCESS.FORCE_REBUILD = True
_C.PREPROCESS.CACHE_DIRNAME = "cache"
_C.PREPROCESS.SEMANTIC = False

# -------------------------
# Solver
# -------------------------
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 1e-5
_C.SOLVER.SEED = 2048

# -------------------------
# Train
# -------------------------
_C.TRAIN = CN()
_C.TRAIN.REG_NORMALIZE = True
_C.TRAIN.REG_LOSS = "MSE"
_C.TRAIN.CLIP_GRAD_NORM = None

# -------------------------
# Result
# -------------------------
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = True

def get_cfg_defaults():
    return _C.clone()
