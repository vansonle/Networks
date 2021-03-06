# module name is image.py

class Image:

    def __init__(self, img):
        self.img = img
        self.img_resized = []
        self.img_rotated_90 = []
        self.img_rotated_180 = []
        self.img_rotated_270 = []
        self.img_flipped_lr = []
        self.img_flipped_ud = []
        self.img_Gaussian_noise = []
        self.Otsu_HoG_overlay = []
        self.watershed_overlay = []
        self.threshold_mask = []
        self.Ncut_segmentation =  []
        self.Ncut_segm_mask =  []
        self.felzen_segmentation = []
        self.felzen_segm_mask =  []
        self.rw_segmentation = []
        self.rw_segm_mask =  []


    def show_img(self):
        img = np.squeeze(self.img)
        width = 10.0
        height = img.shape[0]*width/img.shape[1]
        plt.figure(figsize=(width, height))
        plt.imshow(img)
        plt.show()

    def img_augmentated(self):
        image = np.copy(self.img)
        self.img_rotated_90 = (rotate(image, angle=90, resize=False, center=None,
                     order=1, mode='constant', cval=0, clip=True, preserve_range=False) * 255).astype(np.uint8)
        self.img_rotated_180 = (rotate(image, angle=180, resize=False, center=None,
                     order=1, mode='constant', cval=0, clip=True, preserve_range=False) * 255).astype(np.uint8)
        self.img_rotated_270 = (rotate(image, angle=270, resize=False, center=None,
                     order=1, mode='constant', cval=0, clip=True, preserve_range=False) * 255).astype(np.uint8)
        self.img_flipped_lr = np.fliplr(image)
        self.img_flipped_ud = np.flipud(image)
        self.img_Gaussian_noise = (random_noise(image, mode='gaussian', seed=None, clip=True) * 255).astype(np.uint8)

    def Otsu_HoG_overlay(self, threshold):
        # ---------- Otsu HoG overlay ---------------------------
        image = np.copy(self.img)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,threshold)
        pred_mask =  1 >= thresh

        gx = cv2.Sobel(thresh, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(thresh, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        mag_dim = (mag > 0.5)
        mag_dim = mag_dim.reshape(mag_dim.shape + (1,))
        self.Otsu_HoG_overlay = np.where(mag_dim == True, 255, image)
        # --------------------------------------------------------

    def watershed_overlay(self, threshold):
        # -------- watershed_overlay -----------------------------
        image = np.copy(self.img)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,threshold)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(image,markers)
        image[markers == -1] = [255,0,0]
        self.watershed_overlay = image
        # --------------------------------------------------------


    def threshold_mask(self, threshold):
        # -------- threshold_mask --------------------------------
        img = np.copy(self.img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255, threshold)
        self.threshold_mask =  1 >= thresh
        # --------------------------------------------------------


    def Ncut_segm(self, compact, segm):
        """Function segments an image using the Ncut algorithm
        Parameters: compactness=compact, n_segments=segm
        """

        img = np.copy(self.img)
        labels = segmentation.slic(img, compactness=compact, n_segments=segm)
        g = graph.rag_mean_color(img, labels, mode='similarity')
        labels = graph.cut_normalized(labels, g)
        self.Ncut_segmentation = color.label2rgb(labels, img, kind='avg')

        return self.Ncut_segmentation

    def Ncut_segm_threshold_mask(self, threshold):
        """Creates a mask using threshold
        """

        img = np.copy(self.Ncut_segmentation)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255, threshold)
        self.Ncut_segm_mask =  1 >= thresh

        return self.Ncut_segm_mask
    # --------------------------------------------------------


    def felzen_segm(self, scale, sigma, min_size):
        """Function segments an image using the felzenszwalb algorithm
        """

        img = np.copy(self.img)
        felzen_t = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
        img_segmented_felzen = color.label2rgb(felzen_t, img, kind='avg')

        self.felzen_segmentation = img_segmented_felzen

        return self.felzen_segmentation

    def felzen_segm_threshold_mask(self, threshold):
        """Creates a mask using threshold
        """

        img = np.copy(self.felzen_segmentation)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255, threshold)
        self.felzen_segm_mask =  1 >= thresh

        return self.felzen_segm_mask
    # --------------------------------------------------------


    def rw_segm(self, compact, segm):
        """Function segments an image using the Random Walker algorithm
        Parameters: compactness=compact, n_segments=segm
        """

        img = np.copy(self.img)

        labels = segmentation.slic(img, compactness=compact, n_segments=segm)
        labels += 1  # So that no labelled region is 0 and ignored by regionprops

        rw_t = random_walker(img, labels, beta=130, mode='bf', tol=0.001,
                             copy=True, multichannel=False, return_full_prob=False, spacing=None)
        img_segmented_rw = color.label2rgb(rw_t, img, kind='avg')

        self.rw_segmentation = img_segmented_rw

        return self.rw_segmentation

    def rw_segm_threshold_mask(self, threshold):
        """Creates a mask using threshold
        """

        img = np.copy(self.rw_segmentation)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255, threshold)
        self.rw_segm_mask =  1 >= thresh

        return self.rw_segm_mask
    # --------------------------------------------------------



class ImageResized(Image):
    # resized images
    # Note the CamelCase naming convention used here ;)

    def __init__(self, img, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        self.img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True).astype(np.uint8)
