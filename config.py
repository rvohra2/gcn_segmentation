#DATASET_PATH = '/home/rhythm/notebook/fast-line-drawing-vectorization-master/data/qdraw/'
DATASET_PATH = 'scratch/'
ROOT_PATH = 'scratch/'
OUTPUT_PATH = '/home/rhythm/notebook/vectorData_test/temp/'
CHK_PATH = 'scratch/chk.pt'
MODEL_PATH = 'scratch/model.pt'

#dataset parameters
SPLIT = 'train'
TRANSFORM = True
DATASET = 'baseball'
SEGM = 100
INPUT_WDT = 128
INPUT_HGT = 128

#learning parameters
BATCH_SIZE = 32
EPOCHS = 1000
LR = 1e-5
HIDDEN_LAYER = 1024
OUTPUT_LAYER = 30
NUM_FEAT = 7000
DEVICE = "cuda"


is_L2 = False
is_L1 = True

#GPU parameters
distributed = True
state_file = ""
rank = 0