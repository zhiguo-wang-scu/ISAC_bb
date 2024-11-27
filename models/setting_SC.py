import os

ROOT = '/home/wangzhiguo/ISAC_code_final/user_selection/data_SC/'




DEBUG = False

# DEVICE = 'cpu'
DEVICE = 'cuda'

DAGGER_NUM_TRAIN_EXAMPLES_PER_ITER = 20
DAGGER_NUM_VALID_EXAMPLES_PER_ITER = 20
DAGGER_NUM_ITER = 20
BB_MAX_STEPS = 100

LEARNING_RATE = 0.001

REUSE_DATASET = True


ANTENNA_NFEATS = 3
EDGE_NFEATS = 6
VAR_NFEATS = 13
NODE_DEPTH_INDEX = 7

IN_FEATURES = 219

DATA_PATH = os.path.join(ROOT, 'data/data_multiprocess/gaussian_feature')
MODEL_PATH = os.path.join(ROOT, 'trained_models/gaussian_feature')
RESULT_PATH = os.path.join(ROOT, 'data/gaussian_feature')

LOAD_MODEL = False
LOAD_MODEL_PATH = os.path.join(ROOT, 'user_selection/data_SC/trained_models/gaussian_feature/gnn_iter')

CLASS_IMBALANCE_WT = 11  # for 8,3,5 (N,M,L) use 11, for 12,6,5 max_ant use 11

ETA_EXP = 100000000.0

# This is the weight given to the regularization term. In practice, setting this to 0 also seems to work just as well.
LAMBDA_ETA = 1
