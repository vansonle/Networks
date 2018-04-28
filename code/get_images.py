# I copied below from: https://www.kaggle.com/aglotero/another-iou-metric

# ------------------------------------------------------------
# Gets images and puts them into X_train and Y_train
X_train = []
Y_train = []
print('Getting train images and masks')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]    
    X_train.append(img)
    mask = []
    for i, mask_file in enumerate(next(os.walk(path + '/masks/'))[2]):
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(mask_, axis=-1)
        if np.sum(mask) == 0:
            mask = np.zeros((len(mask_), len(mask_[0]), 1), dtype=np.bool)      
        mask = np.maximum(mask, mask_)
        
    Y_train.append(mask)

X_test = []
print('Getting test images')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    X_test.append(imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS])


print('X_train: {} and length: {}'.format(type(X_train), len(X_train)))
print('Y_train: {} and length: {}'.format(type(Y_train), len(Y_train)))
print('X_test: {} and length: {}'.format(type(X_test), len(X_test)))
# ------------------------------------------------------------
