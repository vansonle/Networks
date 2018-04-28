# Dataframes stage1_train and stage1_test

# ------------------------------------------------------------

# Count the number of masks for each image
number_mask_train = []
for n, id_ in enumerate(stage1_train['ImageId']):
    path = TRAIN_PATH + id_ +'/masks/'
    number_mask_train.append(len(next(os.walk(path))[2]))
stage1_train['No. Mask'] = number_mask_train


# Calculate Otsu
thresh_val_train = []
for i in range(670):
    temp = rgb2gray(X_train[i])
    thresh_val_train.append(threshold_otsu(temp))
stage1_train['Otsu'] = thresh_val_train

thresh_val_test = []
for i in range(65):
    temp = rgb2gray(X_test[i])
    thresh_val_test.append(threshold_otsu(temp))
stage1_test['Otsu'] = thresh_val_test


# Determine Black/White image
BW = []
for i in range(670):
    if X_train[i][3][3][0] == X_train[i][3][3][1] == X_train[i][3][3][2]:
        if X_train[i][20][20][0] == X_train[i][20][20][1] == X_train[i][20][20][2]:
            if X_train[i][50][50][0] == X_train[i][50][50][1] == X_train[i][50][50][2]:
                BW.append(1)
    else:
        BW.append(0)
stage1_train['BW:1, Colour:0'] = BW

BW_test = []
for i in range(65):
    if X_test[i][3][3][0] == X_test[i][3][3][1] == X_test[i][3][3][2]:
        if X_test[i][20][20][0] == X_test[i][20][20][1] == X_test[i][20][20][2]:
            if X_test[i][50][50][0] == X_test[i][50][50][1] == X_test[i][50][50][2]:
                BW_test.append(1)
    else:
        BW_test.append(0)
stage1_test['BW:1, Colour:0'] = BW_test


# Class images
class_images_train = []
for i in range(670):
    if stage1_train['BW:1, Colour:0'].iloc[i] == 0:
        class_images_train.append(0)
    elif stage1_train['BW:1, Colour:0'].iloc[i] == 1 and stage1_train['Otsu'].iloc[i] < 0.5:
        class_images_train.append(1)
    else:
        class_images_train.append(2)
stage1_train['class: colour=0, BW=1, BW/light=2'] = class_images_train

class_images_test = []
for i in range(65):
    if stage1_test['BW:1, Colour:0'].iloc[i] == 0:
        class_images_test.append(0)
    elif stage1_test['BW:1, Colour:0'].iloc[i] == 1 and stage1_test['Otsu'].iloc[i] < 0.5:
        class_images_test.append(1)
    else:
        class_images_test.append(2)
stage1_test['class: colour=0, BW=1, BW/light=2'] = class_images_test


# Dimension of image
dimension_train = []
for i in range(670):
    dimension_train.append(np.shape(X_train[i]))
stage1_train['Dimension'] = dimension_train

dimension_test = []
for i in range(65):
    dimension_test.append(np.shape(X_test[i]))
stage1_test['Dimension'] = dimension_test


# Counting the different classes and BW/colour
df = stage1_train
from prettytable import PrettyTable

values = df['BW:1, Colour:0'].value_counts().keys().tolist()
counts = df['BW:1, Colour:0'].value_counts().tolist()
print('stage1_train: BW:1, Colour:0')
t = PrettyTable(['values', values])
t.add_row(['counts', counts])
print(t)

values = df['class: colour=0, BW=1, BW/light=2'].value_counts().keys().tolist()
counts = df['class: colour=0, BW=1, BW/light=2'].value_counts().tolist()
print('stage1_train: class: colour=0, BW=1, BW/light=2')
t = PrettyTable(['values', values])
t.add_row(['counts', counts])
print(t)

values = df['Dimension'].value_counts().keys().tolist()
counts = df['Dimension'].value_counts().tolist()
print('stage1_train has images with dimension:')
t = PrettyTable()
t.add_column('values', values,  align='c')
t.add_column('counts', counts,  align='c')
print(t)

df = stage1_test
values = df['BW:1, Colour:0'].value_counts().keys().tolist()
counts = df['BW:1, Colour:0'].value_counts().tolist()
print('stage1_test: BW:1, Colour:0')
t = PrettyTable(['values', values])
t.add_row(['counts', counts])
print(t)

values = df['class: colour=0, BW=1, BW/light=2'].value_counts().keys().tolist()
counts = df['class: colour=0, BW=1, BW/light=2'].value_counts().tolist()
print('stage1_test: class: colour=0, BW=1, BW/light=2')
t = PrettyTable(['values', values])
t.add_row(['counts', counts])
print(t)

values = df['Dimension'].value_counts().keys().tolist()
counts = df['Dimension'].value_counts().tolist()
print('stage1_test has images with dimension:')
t = PrettyTable()
t.add_column('values', values,  align='c')
t.add_column('counts', counts,  align='c')
print(t)

# Count the number of masks for each image and updates stage1_train
number_mask_train = []
for n, id_ in enumerate(stage1_train['ImageId']):
    path = TRAIN_PATH + id_ +'/masks/'
    number_mask_train.append(len(next(os.walk(path))[2]))
stage1_train['No. Mask'] = number_mask_train


# sizes_test stores the dimension of the test images in a list object
# sizes_test is needed later
sizes_test = []
for i in range(len(stage1_test['Dimension'])):
    sizes_test.append([stage1_test['Dimension'][i][0], stage1_test['Dimension'][i][1]])


# ------------------------------------------------------------
print('stage1_train: {}'.format(type(stage1_train)))
print('stage1_test: {}'.format(type(stage1_test)))
# ------------------------------------------------------------
