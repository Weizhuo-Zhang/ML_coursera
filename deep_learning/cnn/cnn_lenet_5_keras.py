def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.

    ######### LeNet - 5 #########
    X_input = Input(input_shape)

    ### Conv 0 ###
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(6, (5, 5), strides = (1, 1), name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('tanh')(X)
    # AVGPOOL
    X = AveragePooling2D((2, 2), name='avg_pool0')(X)

    ### Conv 1 ###
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(16, (5, 5), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('tanh')(X)
    # AVGPOOL
    X = AveragePooling2D((2, 2), name='avg_pool1')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(120, activation='tanh', name='fc1')(X)
    X = Dense(84, activation='tanh', name='fc2')(X)
    X = Dense(1, activation='sigmoid', name='lenet_5')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    ######### LeNet - 5 #########

    ### END CODE HERE ###

    return model
