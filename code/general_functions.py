def P(object):
    # Returns the most used information
    shape = np.shape(object)
    type_object = type(object)
    len_object = len(object)
    return print('shape: {}, type: {}'.format(shape, type_object))

def show_img(img):
    width = 10.0
    height = img.shape[0]*width/img.shape[1]
    plt.figure(figsize=(width, height))
    plt.imshow(np.squeeze(img))
    plt.show()

def show_multi_img(images, m, n, height, width):
    # Function plots multiple images of an array
    # images need to be in one array with the length of the array
    # equal to the number of images
    # m, n determine the number of rows and columns

    # if multiple images put them together like this:
    # images = [image1, image2, image3]
    plt.figure(figsize=(width, height))
    for i in range(len(images)):
        plt.subplot(m,n,i+1)
        plt.imshow(images[i], aspect='auto')
        plt.axis('off')
        plt.title('Image {}'.format(i))
    plt.show()

def unique_values_count(array):
    # Takes an array and flattens it
    # Then counts the unique values
    # Then counts unique values
    df = pd.DataFrame()
    df['temp_label'] = array.flatten()
    values = df['temp_label'].value_counts().keys().tolist()
    counts = df['temp_label'].value_counts().tolist()
    return values, counts

def print_obj(object):
    for i in object:
        print(i)
        print(object[i])

def index_value(value, column_name, df):
    index = df.index[df[column_name] == value].tolist()
    return index[0]

def vlookup(value, column_name_value, column_name_lookup, df):
    index = df.index[df[column_name_value] == value].tolist()
    lookup_value = df[column_name_lookup][index[0]]
    return lookup_value
