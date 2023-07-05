import os
from os.path import isfile, join

import cv2
import numpy as np
import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, \
    Dense, Flatten, Conv2D, Lambda, Input, BatchNormalization, Activation
from tensorflow.keras.optimizers import schedules, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tqdm import tqdm
from config import ConstantConfig

encoder = ConstantConfig.ENCODER
train_data_path = r"C:\\Users\\annch\\OneDrive\\Desktop\\master\\ხელნაწერები"


def model_configuration():
    """
        Get configuration variables for the model.
    """

    # Load dataset for computing dataset size
    (input_train, _), (_, _) = load_dataset()

    # Generic config
    width, height, channels = 32, 32, 3
    batch_size = 128
    num_classes = 33
    validation_split = 0.1  # 45/5 per the He et al. paper
    verbose = 1
    n = 3
    init_fm_dim = 16
    shortcut_type = "identity"  # or: projection

    # Dataset size
    train_size = (1 - validation_split) * len(input_train)
    val_size = (validation_split) * len(input_train)

    # Number of steps per epoch is dependent on batch size
    maximum_number_iterations = 64000  # per the He et al. paper
    steps_per_epoch = tensorflow.math.floor(train_size / batch_size)
    val_steps_per_epoch = tensorflow.math.floor(val_size / batch_size)
    epochs = tensorflow.cast(tensorflow.math.floor(maximum_number_iterations / steps_per_epoch), \
                             dtype=tensorflow.int64)

    # Define loss function
    loss = tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Learning rate config per the He et al. paper
    boundaries = [32000, 48000]
    values = [0.1, 0.01, 0.001]
    lr_schedule = schedules.PiecewiseConstantDecay(boundaries, values)

    # Set layer init
    initializer = tensorflow.keras.initializers.HeNormal()

    # Define optimizer
    optimizer_momentum = 0.9
    optimizer_additional_metrics = ["accuracy"]
    optimizer = SGD(learning_rate=lr_schedule, momentum=optimizer_momentum)

    # Load Tensorboard callback
    tensorboard = TensorBoard(
        log_dir=os.path.join(os.getcwd(), "logs"),
        histogram_freq=1,
        write_images=True
    )

    # Save a model checkpoint after every epoch
    checkpoint = ModelCheckpoint(
        os.path.join(os.getcwd(), "model_checkpoint"),
        save_freq="epoch"
    )

    # Add callbacks to list
    callbacks = [
        tensorboard,
        checkpoint
    ]

    # Create config dictionary
    config = {
        "width": width,
        "height": height,
        "dim": channels,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "validation_split": validation_split,
        "verbose": verbose,
        "stack_n": n,
        "initial_num_feature_maps": init_fm_dim,
        "training_ds_size": train_size,
        "steps_per_epoch": steps_per_epoch,
        "val_steps_per_epoch": val_steps_per_epoch,
        "num_epochs": epochs,
        "loss": loss,
        "optim": optimizer,
        "optim_learning_rate_schedule": lr_schedule,
        "optim_momentum": optimizer_momentum,
        "optim_additional_metrics": optimizer_additional_metrics,
        "initializer": initializer,
        "callbacks": callbacks,
        "shortcut_type": shortcut_type
    }

    return config


def load_dataset():
    """
        Load the CIFAR-10 dataset
    """
    # return cifar10.load_data()
    img_data = {}
    labels, img_vecs = [], []
    test_img_vecs, test_labels = [], []
    for f in tqdm(os.listdir(train_data_path)):
        if isfile(join(train_data_path, f)) or f == 'board':
            continue
        dir_path = join(train_data_path, f)
        for imgs in os.listdir(dir_path):
            path = join(dir_path, imgs)
            img_vector = cv2.imdecode(np.fromfile(
                path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE).astype("float32")
            img_vector = cv2.resize(
                img_vector, (50, 50), interpolation=cv2.INTER_AREA) / 255
            img_vector = np.expand_dims(img_vector, -1)
            key = [c for c in imgs if not 0 <= ord(c) <= 127][0]
            key = encoder[key]
            if img_data.get(key, 0) < 100:
                test_img_vecs.append(img_vector)
                test_labels.append(np.array([key]))
                img_data[key] = img_data.get(key, 0) + 1
            else:
                img_vecs.append(img_vector)
                labels.append(np.array([key]))
    return (np.array(img_vecs), np.array(labels)), (np.array(test_img_vecs), np.array(test_labels))


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    # SOURCE: https://jkjung-avt.github.io/keras-image-cropping/
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    SOURCE: https://jkjung-avt.github.io/keras-image-cropping/
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops, batch_y


def preprocessed_dataset():
    """
        Load and preprocess the CIFAR-10 dataset.
    """
    (input_train, target_train), (input_test, target_test) = load_dataset()

    # Retrieve shape from model configuration and unpack into components
    config = model_configuration()
    width, height, dim = config.get("width"), config.get("height"), \
        config.get("dim")
    num_classes = config.get("num_classes")

    # Data augmentation: perform zero padding on datasets
    paddings = tensorflow.constant([[0, 0, ], [4, 4], [4, 4], [0, 0]])
    input_train = tensorflow.pad(input_train, paddings, mode="CONSTANT")

    # Convert scalar targets to categorical ones
    target_train = tensorflow.keras.utils.to_categorical(target_train)
    target_test = tensorflow.keras.utils.to_categorical(target_test)

    # Data generator for training data
    train_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        validation_split=config.get("validation_split"),
        horizontal_flip=True,
        rescale=1. / 255,
        preprocessing_function=tensorflow.keras.applications.resnet50.preprocess_input
    )

    # Generate training and validation batches
    train_batches = train_generator.flow(input_train, target_train, batch_size=config.get("batch_size"),
                                         subset="training")
    validation_batches = train_generator.flow(input_train, target_train, batch_size=config.get("batch_size"),
                                              subset="validation")
    train_batches = crop_generator(train_batches, config.get("height"))
    validation_batches = crop_generator(validation_batches, config.get("height"))

    # Data generator for testing data
    test_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tensorflow.keras.applications.resnet50.preprocess_input,
        rescale=1. / 255)

    # Generate test batches
    test_batches = test_generator.flow(input_test, target_test, batch_size=config.get("batch_size"))

    return train_batches, validation_batches, test_batches


def residual_block(x, number_of_filters, match_filter_size=False):
    """
        Residual block with
    """
    # Retrieve initializer
    config = model_configuration()
    initializer = config.get("initializer")

    # Create skip connection
    x_skip = x

    # Perform the original mapping
    if match_filter_size:
        x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(2, 2),
                   kernel_initializer=initializer, padding="same")(x_skip)
    else:
        x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(1, 1),
                   kernel_initializer=initializer, padding="same")(x_skip)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = Conv2D(number_of_filters, kernel_size=(3, 3),
               kernel_initializer=initializer, padding="same")(x)
    x = BatchNormalization(axis=3)(x)

    # Perform matching of filter numbers if necessary
    if match_filter_size and config.get("shortcut_type") == "identity":
        x_skip = Lambda(lambda x: tensorflow.pad(x[:, ::2, ::2, :], tensorflow.constant(
            [[0, 0, ], [0, 0], [0, 0], [number_of_filters // 4, number_of_filters // 4]]), mode="CONSTANT"))(x_skip)
    elif match_filter_size and config.get("shortcut_type") == "projection":
        x_skip = Conv2D(number_of_filters, kernel_size=(1, 1),
                        kernel_initializer=initializer, strides=(2, 2))(x_skip)

    # Add the skip connection to the regular mapping
    x = Add()([x, x_skip])

    # Nonlinearly activate the result
    x = Activation("relu")(x)

    # Return the result
    return x


def ResidualBlocks(x):
    """
        Set up the residual blocks.
    """
    # Retrieve values
    config = model_configuration()

    # Set initial filter size
    filter_size = config.get("initial_num_feature_maps")

    # Paper: "Then we use a stack of 6n layers (...)
    #	with 2n layers for each feature map size."
    # 6n/2n = 3, so there are always 3 groups.
    for layer_group in range(3):

        # Each block in our code has 2 weighted layers,
        # and each group has 2n such blocks,
        # so 2n/2 = n blocks per group.
        for block in range(config.get("stack_n")):

            # Perform filter size increase at every
            # first layer in the 2nd block onwards.
            # Apply Conv block for projecting the skip
            # connection.
            if layer_group > 0 and block == 0:
                filter_size *= 2
                x = residual_block(x, filter_size, match_filter_size=True)
            else:
                x = residual_block(x, filter_size)

    # Return final layer
    return x


def model_base(shp):
    """
        Base structure of the model, with residual blocks
        attached.
    """
    # Get number of classes from model configuration
    config = model_configuration()
    initializer = model_configuration().get("initializer")

    # Define model structure
    # logits are returned because Softmax is pushed to loss function.
    inputs = Input(shape=shp)
    x = Conv2D(config.get("initial_num_feature_maps"), kernel_size=(3, 3),
               strides=(1, 1), kernel_initializer=initializer, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = ResidualBlocks(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(32,kernel_initializer=initializer)(x)

    return inputs, outputs


def init_model():
    """
        Initialize a compiled ResNet model.
    """
    # Get shape from model configuration
    config = model_configuration()

    # Get model base
    inputs, outputs = model_base((config.get("width"), config.get("height"),
                                  config.get("dim")))

    # Initialize and compile model
    model = Model(inputs, outputs, name=config.get("name"))
    model.compile(loss=config.get("loss"), optimizer=config.get("optim"),
                  metrics=config.get("optim_additional_metrics"))

    # Print model summary
    model.summary()

    return model


def train_model(model, train_batches, validation_batches):
    """
        Train an initialized model.
    """

    # Get model configuration
    config = model_configuration()

    # Fit data to model
    model.fit(train_batches,
              batch_size=config.get("batch_size"),
              epochs=config.get("num_epochs"),
              verbose=config.get("verbose"),
              callbacks=config.get("callbacks"),
              steps_per_epoch=config.get("steps_per_epoch"),
              validation_data=validation_batches,
              validation_steps=config.get("val_steps_per_epoch"))

    return model


def evaluate_model(model, test_batches):
    """
        Evaluate a trained model.
    """
    # Evaluate model
    score = model.evaluate(test_batches, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


def training_process():
    """
        Run the training process for the ResNet model.
    """

    # Get dataset
    train_batches, validation_batches, test_batches = preprocessed_dataset()

    # Initialize ResNet
    resnet = init_model()

    # Train ResNet model
    trained_resnet = train_model(resnet, train_batches, validation_batches)

    # Evalute trained ResNet model post training
    evaluate_model(trained_resnet, test_batches)


if __name__ == "__main__":
    training_process()
