# ------------------------------------------------------------
# This is my own code
# Crop

def crop_array(array_crop):

    # Crops the images

    array_cropped = []

    for k in range(len(array_crop)):

        if np.shape(array_crop[k]) == (520, 696, 3) or np.shape(array_crop[k]) == (520, 696, 1):
            # (520, 696, 3)
            x = [0, 260, 520]
            y = [0, 348, 696]
            for i in range(2):
                for j in range(2):
                    array_cropped.append(array_crop[k][x[i]:x[i+1], y[j]:y[j+1]])

        elif np.shape(array_crop[k]) == (512, 640, 3) or np.shape(array_crop[k]) == (512, 640, 1):
            # (512, 640, 3)
            x = [0, 256, 512]
            y = [0, 320, 640]
            for i in range(2):
                for j in range(2):
                    array_cropped.append(array_crop[k][x[i]:x[i+1], y[j]:y[j+1]])

        elif np.shape(array_crop[k]) == (603, 1272, 3) or np.shape(array_crop[k]) == (603, 1272, 1):
            # (603, 1272, 3)
            x = [0, 300, 603]
            y = [0, 318, 636, 954, 1272]
            for i in range(2):
                for j in range(4):
                    array_cropped.append(array_crop[k][x[i]:x[i+1], y[j]:y[j+1]])

        elif np.shape(array_crop[k]) == (1024, 1024, 3) or np.shape(array_crop[k]) == (1024, 1024, 1):
            # (1024, 1024, 3)
            x = [0, 256, 512, 768, 1024]
            y = [0, 256, 512, 768, 1024]
            for i in range(4):
                for j in range(4):
                    array_cropped.append(array_crop[k][x[i]:x[i+1], y[j]:y[j+1]])

        elif np.shape(array_crop[k]) == (1040, 1388, 3) or np.shape(array_crop[k]) == (1040, 1388, 1):
            # (1024, 1388, 3)
            x = [0, 260, 520, 780, 1040]
            y = [0, 277, 554, 831, 1108, 1388]
            for i in range(4):
                for j in range(5):
                    array_cropped.append(array_crop[k][x[i]:x[i+1], y[j]:y[j+1]])

    return array_cropped

X_cropped = crop_array(X_train)
Y_cropped = crop_array(Y_train)

print('X_cropped: {} and shape: {}'.format(type(X_cropped), np.shape(X_cropped)))
print('Y_cropped: {} and shape: {}'.format(type(Y_cropped), np.shape(Y_cropped)))
# ------------------------------------------------------------
