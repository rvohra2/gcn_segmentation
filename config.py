DATASET_PATH = '/home/rhythm/notebook/fast-line-drawing-vectorization-master/data/qdraw/'
ROOT_PATH = '/home/rhythm/notebook/vectorData_test/temp/processed/'
OUTPUT_PATH = '/home/rhythm/notebook/vectorData_test/temp/'
CHK_PATH = '/home/rhythm/notebook/vectorData_test/temp/chk.pt'

#dataset parameters
SPLIT = 'train'
TRANSFORM = False
DATASET = 'cat'

#learning parameters
BATCH_SIZE = 1
EPOCHS = 200
LR = 1e-5
HIDDEN_LAYER = 1024
OUTPUT_LAYER = 30

#GPU parameters
distributed = True
state_file = ""
rank = 0