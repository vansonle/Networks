class model:
    # Neural Network models

    def __init__(self):

        self.model_UNet_large_filter = []
        self.model_UNet_large = []
        self.model_UNet = []

        self.model_MLP_hidden_layer_1 = []
        self.model_MLP_hidden_layer_2 = []
        self.model_MLP_hidden_layer_3 = []
        self.model_MLP_hidden_layer_9 = []
        self.model_MLP_hidden_layer_20 = []
        self.model_MLP_hidden_layer_20_l1 = []
        self.model_MLP_hidden_layer_20_dropout = []
        self.model_Simple_MLP = []

        self.model_Simple_CNN = []
        self.model_CNN_upsize = []
        self.model_CNN_downsize = []
        self.model_CNN_downsize_large = []
        self.model_CNN_upsize_MaxPooling = []


    def UNet_large_filter(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Conv2D(8, (9, 9), activation='relu', padding='same') (s)
        c1 = Conv2D(8, (9, 9), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='tanh') (c9)

        self.model_UNet_large_filter = Model(inputs=[inputs], outputs=[outputs])
        self.model_UNet_large_filter.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def UNet_large(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (s)
        c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(256, (3, 3), activation='relu', padding='same') (p4)
        c5 = Conv2D(256, (3, 3), activation='relu', padding='same') (c5)

        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (c6)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(64, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(32, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(16, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        self.model_UNet_large = Model(inputs=[inputs], outputs=[outputs])
        self.model_UNet_large.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def UNet(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
        c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

        u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
        c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

        u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
        c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='tanh') (c9)

        self.model_UNet = Model(inputs=[inputs], outputs=[outputs])
        self.model_UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def MLP_hidden_layer_1(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Dense(2048, activation='relu') (s)
        outputs = Dense(1, activation='tanh') (c1)

        self.model_MLP_hidden_layer_1 = Model(inputs=[inputs], outputs=[outputs])
        self.model_MLP_hidden_layer_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def MLP_hidden_layer_2(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Dense(128, activation='relu') (s)
        c2 = Dense(64, activation='relu') (c1)
        outputs = Dense(1, activation='tanh') (c2)

        self.model_MLP_hidden_layer_2 = Model(inputs=[inputs], outputs=[outputs])
        self.model_MLP_hidden_layer_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def MLP_hidden_layer_3(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Dense(128, activation='relu') (s)
        c2 = Dense(64, activation='relu') (c1)
        c3 = Dense(64, activation='relu') (c2)
        outputs = Dense(1, activation='tanh') (c3)

        self.model_MLP_hidden_layer_3 = Model(inputs=[inputs], outputs=[outputs])
        self.model_MLP_hidden_layer_3.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def MLP_hidden_layer_9(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Dense(32, activation='relu') (s)
        c2 = Dense(32, activation='relu') (c1)
        c3 = Dense(32, activation='relu') (c2)
        c4 = Dense(32, activation='relu') (c3)
        c5 = Dense(32, activation='relu') (c4)
        c6 = Dense(32, activation='relu') (c5)
        c7 = Dense(32, activation='relu') (c6)
        c8 = Dense(32, activation='relu') (c7)
        c9 = Dense(32, activation='relu') (c8)

        outputs = Dense(1, activation='tanh') (c9)

        self.model_MLP_hidden_layer_9 = Model(inputs=[inputs], outputs=[outputs])
        self.model_MLP_hidden_layer_9.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def MLP_hidden_layer_20(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Dense(16, activation='relu') (s)
        c2 = Dense(16, activation='relu') (c1)
        c3 = Dense(16, activation='relu') (c2)
        c4 = Dense(16, activation='relu') (c3)
        c5 = Dense(16, activation='relu') (c4)
        c6 = Dense(16, activation='relu') (c5)
        c7 = Dense(16, activation='relu') (c6)
        c8 = Dense(16, activation='relu') (c7)
        c9 = Dense(16, activation='relu') (c8)
        c10 = Dense(16, activation='relu') (c9)

        c11 = Dense(16, activation='relu') (c10)
        c12 = Dense(16, activation='relu') (c11)
        c13 = Dense(16, activation='relu') (c12)
        c14 = Dense(16, activation='relu') (c13)
        c15 = Dense(16, activation='relu') (c14)
        c16 = Dense(16, activation='relu') (c15)
        c17 = Dense(16, activation='relu') (c16)
        c18 = Dense(16, activation='relu') (c17)
        c19 = Dense(16, activation='relu') (c18)
        c20 = Dense(16, activation='relu') (c19)

        outputs = Dense(1, activation='tanh') (c20)

        self.model_MLP_hidden_layer_20 = Model(inputs=[inputs], outputs=[outputs])
        self.model_MLP_hidden_layer_20.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def MLP_hidden_layer_20_l1(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Dense(16, kernel_regularizer='l1', activation='relu') (s)
        c2 = Dense(16, kernel_regularizer='l1', activation='relu') (c1)
        c3 = Dense(16, kernel_regularizer='l1', activation='relu') (c2)
        c4 = Dense(16, kernel_regularizer='l1', activation='relu') (c3)
        c5 = Dense(16, kernel_regularizer='l1', activation='relu') (c4)
        c6 = Dense(16, kernel_regularizer='l1', activation='relu') (c5)
        c7 = Dense(16, kernel_regularizer='l1', activation='relu') (c6)
        c8 = Dense(16, kernel_regularizer='l1', activation='relu') (c7)
        c9 = Dense(16, kernel_regularizer='l1', activation='relu') (c8)
        c10 = Dense(16, kernel_regularizer='l1', activation='relu') (c9)

        c11 = Dense(16, kernel_regularizer='l1', activation='relu') (c10)
        c12 = Dense(16, kernel_regularizer='l1', activation='relu') (c11)
        c13 = Dense(16, kernel_regularizer='l1', activation='relu') (c12)
        c14 = Dense(16, kernel_regularizer='l1', activation='relu') (c13)
        c15 = Dense(16, kernel_regularizer='l1', activation='relu') (c14)
        c16 = Dense(16, kernel_regularizer='l1', activation='relu') (c15)
        c17 = Dense(16, kernel_regularizer='l1', activation='relu') (c16)
        c18 = Dense(16, kernel_regularizer='l1', activation='relu') (c17)
        c19 = Dense(16, kernel_regularizer='l1', activation='relu') (c18)
        c20 = Dense(16, kernel_regularizer='l1', activation='relu') (c19)

        outputs = Dense(1, activation='tanh') (c20)

        self.model_MLP_hidden_layer_20_l1 = Model(inputs=[inputs], outputs=[outputs])
        self.model_MLP_hidden_layer_20_l1.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def MLP_hidden_layer_20_dropout(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Dense(16, activation='relu') (s)
        c2 = Dense(16, activation='relu') (c1)
        c3 = Dense(16, activation='relu') (c2)
        c4 = Dense(16, activation='relu') (c3)
        c5 = Dense(16, activation='relu') (c4)
        d1 = Dropout(0.2) (c5)

        c6 = Dense(16, activation='relu') (d1)
        c7 = Dense(16, activation='relu') (c6)
        c8 = Dense(16, activation='relu') (c7)
        c9 = Dense(16, activation='relu') (c8)
        c10 = Dense(16, activation='relu') (c9)
        d2 = Dropout(0.2) (c10)

        c11 = Dense(16, activation='relu') (d2)
        c12 = Dense(16, activation='relu') (c11)
        c13 = Dense(16, activation='relu') (c12)
        c14 = Dense(16, activation='relu') (c13)
        c15 = Dense(16, activation='relu') (c14)
        d3 = Dropout(0.2) (c15)

        c16 = Dense(16, activation='relu') (d3)
        c17 = Dense(16, activation='relu') (c16)
        c18 = Dense(16, activation='relu') (c17)
        c19 = Dense(16, activation='relu') (c18)
        c20 = Dense(16, activation='relu') (c19)
        d4 = Dropout(0.2) (c20)

        outputs = Dense(1, activation='tanh') (d4)

        self.model_MLP_hidden_layer_20_dropout = Model(inputs=[inputs], outputs=[outputs])
        self.model_MLP_hidden_layer_20_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def Simple_MLP(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Dense(64, input_dim=10, activation='relu') (s)
        c1 = Dense(64, input_dim=10, activation='relu') (c1)

        c2 = Dense(64, input_dim=10, activation='relu') (c1)
        c2 = Dense(64, input_dim=10, activation='relu') (c2)

        c3 = Dense(64, input_dim=10, activation='relu') (c2)
        c3 = Dense(64, input_dim=10, activation='relu') (c3)

        outputs = Dense(1, activation='tanh') (c3)

        self.model_Simple_MLP = Model(inputs=[inputs], outputs=[outputs])
        self.model_Simple_MLP.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def Simple_CNN(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
        c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)

        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c1)
        c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)

        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c3)

        self.model_Simple_CNN = Model(inputs=[inputs], outputs=[outputs])
        self.model_Simple_CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def CNN_upsize(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Conv2D(2, (3, 3), activation='relu', padding='same') (s)
        c2 = Conv2D(4, (3, 3), activation='relu', padding='same') (c1)
        c3 = Conv2D(8, (3, 3), activation='relu', padding='same') (c2)
        c4 = Conv2D(16, (3, 3), activation='relu', padding='same') (c3)
        c5 = Conv2D(32, (3, 3), activation='relu', padding='same') (c4)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
        c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (c6)

        #p3 = MaxPooling2D((2, 2)) (c2)

        outputs = Conv2D(1, (1, 1), activation='tanh') (c7)

        self.model_CNN_upsize = Model(inputs=[inputs], outputs=[outputs])
        self.model_CNN_upsize.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def CNN_downsize(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Conv2D(128, (3, 3), activation='relu', padding='same') (s)
        c2 = Conv2D(64, (3, 3), activation='relu', padding='same') (c1)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
        c4 = Conv2D(16, (3, 3), activation='relu', padding='same') (c3)
        c5 = Conv2D(8, (3, 3), activation='relu', padding='same') (c4)
        c6 = Conv2D(4, (3, 3), activation='relu', padding='same') (c5)
        c7 = Conv2D(2, (3, 3), activation='relu', padding='same') (c6)

        #p3 = MaxPooling2D((2, 2)) (c2)

        outputs = Conv2D(1, (1, 1), activation='tanh') (c7)

        self.model_CNN_downsize = Model(inputs=[inputs], outputs=[outputs])
        self.model_CNN_downsize.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def CNN_downsize_large(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Conv2D(256, (3, 3), activation='relu', padding='same') (s)
        c2 = Conv2D(64, (3, 3), activation='relu', padding='same') (c1)
        c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
        c4 = Conv2D(16, (3, 3), activation='relu', padding='same') (c3)
        c5 = Conv2D(8, (3, 3), activation='relu', padding='same') (c4)
        c6 = Conv2D(4, (3, 3), activation='relu', padding='same') (c5)
        c7 = Conv2D(2, (3, 3), activation='relu', padding='same') (c6)

        #p3 = MaxPooling2D((2, 2)) (c2)

        outputs = Conv2D(1, (1, 1), activation='tanh') (c7)

        self.model_CNN_downsize_large = Model(inputs=[inputs], outputs=[outputs])
        self.model_CNN_downsize_large.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])

    def CNN_upsize_MaxPooling(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        K.clear_session()
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x ) (inputs)

        c1 = Conv2D(2, (3, 3), activation='relu', padding='same') (s)
        c2 = Conv2D(4, (3, 3), activation='relu', padding='same') (c1)
        c3 = Conv2D(8, (3, 3), activation='relu', padding='same') (c2)
        c4 = Conv2D(16, (3, 3), activation='relu', padding='same') (c3)
        c5 = Conv2D(32, (3, 3), activation='relu', padding='same') (c4)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
        c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (c6)

        p3 = MaxPooling2D((2, 2)) (c7)
        u6 = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same') (p3)

        outputs = Conv2D(1, (1, 1), activation='tanh') (u6)

        self.model_CNN_upsize_MaxPooling = Model(inputs=[inputs], outputs=[outputs])
        self.model_CNN_upsize_MaxPooling.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])
