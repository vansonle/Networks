# Setting up the data, parameters and put them into arrays or dataframes
# I copied below from: https://www.kaggle.com/aglotero/another-iou-metric


# ------------------------------------------------------------
# Set some parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
#seed = 99
#random.seed = seed
#np.random.seed = seed
print('Global parameters: IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH, TEST_PATH')
# ------------------------------------------------------------
