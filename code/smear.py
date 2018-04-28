# ------------------------------------------------------------
# Gets images and puts them into X_smear and Y_smear

SMEAR_PATH = '../input/smears/'
smear_ids = next(os.walk(SMEAR_PATH))[1]

X_smear = []
Y_smear = []

print('Getting smear images and masks')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(smear_ids), total=len(smear_ids)):
    path = SMEAR_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    X_smear.append(img)
    mask = []
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(mask_, axis=-1)
        if np.sum(mask) == 0:
            mask = np.zeros((len(mask_), len(mask_[0]), 1), dtype=np.bool)
        mask = np.maximum(mask, mask_)
    Y_smear.append(mask)

print('X_smear: {} and shape: {}'.format(type(X_smear), np.shape(X_smear)))
print('Y_smear: {} and shape: {}'.format(type(Y_smear), np.shape(Y_smear)))
# ------------------------------------------------------------
