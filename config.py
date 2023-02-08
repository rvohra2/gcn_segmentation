#DATASET_PATH = '/home/rhythm/notebook/fast-line-drawing-vectorization-master/data/qdraw/'
DATASET_PATH = '/media/storage/'
ROOT_PATH = '/home/rhythm/notebook/vectorData_test/temp/processed/'
OUTPUT_PATH = '/home/rhythm/notebook/vectorData_test/temp/processed/'
CHK_PATH = '/home/rhythm/notebook/vectorData_test/temp/processed/chk.pt'
MODEL_PATH = '/home/rhythm/notebook/vectorData_test/temp/processed/model.pt'

#dataset parameters
SPLIT = 'test'
TRANSFORM = False
DATASET = 'baseball'
SEGM = 100
INPUT_WDT = 128
INPUT_HGT = 128

#learning parameters
BATCH_SIZE = 1
EPOCHS = 500
LR = 1e-5
HIDDEN_LAYER = 1024
OUTPUT_LAYER = 3
NUM_FEAT = 7000
DEVICE = "cuda"


is_L2 = False
is_L1 = True

#GPU parameters
distributed = True
state_file = ""
rank = 0