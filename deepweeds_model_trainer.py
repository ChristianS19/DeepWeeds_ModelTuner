# Import Packages
try:
    # import argparse
    import os
    from datetime import datetime
    import time
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow_datasets as tfds
    import tensorflow_addons as tfa
    from tensorflow.keras import Model, layers
    from tensorflow.keras.optimizers.legacy import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
    from tensorflow.keras.applications import ResNet50V2, InceptionV3, Xception, VGG16
    from tensorflow.keras.applications import MobileNetV2, DenseNet121, EfficientNetV2B1, ConvNeXtTiny
    from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess_input
    from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
    from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess_input
    from tensorflow.keras.applications.convnext import preprocess_input as cnt_preprocess_input
    print("All modules imported\n")
except BaseException as exception:
    print(f"An exception in importing occurred: {exception}")

#     # Check Required Versions
print("TF version:", tf.__version__)
print("Number GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Number CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
print("Tensorflow Datasets version:", tfds.__version__)


# Set Functions & classes
class ModelDict:
    def __init__(self):
        self.model_dict = {
            1: ['ResNet50V2', ResNet50V2, (224, 224), resnetv2_preprocess_input],
            2: ['InceptionV3', InceptionV3, (299, 299), inception_preprocess_input],
            3: ['Xception', Xception, (299, 299), xception_preprocess_input],
            4: ['VGG16', VGG16, (224, 224), vgg16_preprocess_input],
            5: ['MobileNetV2', MobileNetV2, (224, 224), mobilenet_v2_preprocess_input],
            6: ['DenseNet121', DenseNet121, (224, 224), densenet_preprocess_input],
            7: ['EfficientNetV2B1', EfficientNetV2B1, (240, 240), efficientnet_preprocess_input],
        }
        self._models = {}

    def __getitem__(self, key):
        if key not in self._models:
            model_name, model_fn, img_size, preprocess_input = self.model_dict[key]
            base_model = model_fn(include_top=False, weights='imagenet')
            self._models[key] = (model_name, base_model, preprocess_input, img_size)
        return self._models[key][:3], self._models[key][3]

    def select_model(self):
        print("Available models:")
        for key, model_info in self.model_dict.items():
            print(f"{key}: {model_info[0]}")
        while True:
            # selection = int(input("Please select a model by entering its number, or enter '0' to quit: "))
            try:
                if selection == 0:
                    return exit_call()
                if selection in self.model_dict:
                    print(f"Selected model: {self.model_dict[selection][0]}")
                    return self[selection]
                else:
                    print("Model not available. Select a valid model number.")
            except ValueError:
                print("Invalid input. Please enter a valid model number.")


def create_folders(model_name):
    """Creates a Model folder (if not existing) and subfolders for saved models and outputs."""
    # script_path=os.getcwd()
    now = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{model_name}_{now}"
    save_path = f'./{folder_name}/'
    models_path = f'{save_path}/models/'
    output_path = f'{save_path}/outputs/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(models_path)
        os.makedirs(output_path)

    return folder_name, save_path, models_path, output_path


def load_model_local(model_file_path):
    if not os.path.isfile(model_file_path):
        print("Error: That is not a valid filepath or the file does not exist, try again...")
    else:
        try:
            model = keras.models.load_model(model_file_path)
        except BaseException as exception:
            print(f"An exception occurred: {exception}")
    return model


def get_folders(model_file_path):
    model_name = os.path.basename(model_file_path).split('_')[0]
    parent_dir = os.path.dirname(os.path.dirname(model_file_path))
    models_folder = 'models'
    outputs_folder = 'outputs'

    # Check if folders exist
    models_path = os.path.join(parent_dir, models_folder)
    outputs_path = os.path.join(parent_dir, outputs_folder)

    counter = 1
    while os.path.exists(models_path) or os.path.exists(outputs_path):
        models_folder = f'models_{counter}'
        outputs_folder = f'outputs_{counter}'
        models_path = os.path.join(parent_dir, models_folder)
        outputs_path = os.path.join(parent_dir, outputs_folder)
        counter += 1

    # Create the new folders
    os.makedirs(models_path)
    os.makedirs(outputs_path)

    return model_name, f'{models_path}/', f'{outputs_path}/'


def load_data(data_name, splits=None):
    """Load and Split Data from Tensorflow Datasets"""
    (train_ds, validation_ds, test_ds), info = tfds.load(
        data_name,
        # data_dir="./data/",
        split=splits,
        shuffle_files=True,
        as_supervised=True,  # Include labels
        with_info=True)

    return (train_ds, validation_ds, test_ds), info


def data_description():
    """Returns information on dataset."""
    class_names = info.features["label"].names
    n_classes = info.features["label"].num_classes
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    img_shape = info.features["image"].shape

    # print('Labels and Names of Classes: ')
    # for i in range(len(class_labels)):
    #     print(class_labels[i], ':', class_names[i])
    print(f'\nShape of images: {img_shape}\n')
    print(f'Number of training samples: {len(train_ds)}')
    print(f'Number of validation samples: {len(validation_ds)}')
    print(f'Number of test samples: {len(test_ds)}')

    return class_names, n_classes, class_labels, img_shape


def split_class_counts(ds):
    """ Returns:
    A list of counts for each class in a dataset
    A list of class labels for dataset
    """
    labels_list = []
    for images, labels in ds:
        labels_list.append(labels.numpy())  # Convert tensor to numpy array
    labels_counts = pd.Series(labels_list).value_counts().sort_index()

    return labels_list, labels_counts


def gen_ds_nums(ds1, ds2, ds3):
    """Outputs a .csv containing information on train, validation, test datasets"""
    train_labels, train_counts = split_class_counts(ds1)
    val_labels, val_counts = split_class_counts(ds2)
    test_labels, test_counts = split_class_counts(ds3)

    df = pd.DataFrame({
        'class_name': class_names,
        'class_label': class_labels,
        '# samples(train)': train_counts,
        '# samples(validation)': val_counts,
        '# samples(test)': test_counts,
        'proportion(train)': round(train_counts / len(train_ds), 2),
        'proportion(validation)': round(val_counts / len(validation_ds), 2),
        'proportion(test)': round(test_counts / len(test_ds), 2),
    })
    return df


def save_df(df):
    return df.to_csv(f'{output_path}dataset_information_and_splits.csv', index=False)


def calculate_class_weights(ds):
    """ Returns a dictionary of the class weights for dataset."""
    from sklearn.utils import class_weight
    ds_labels_list, ds_counts = split_class_counts(ds)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(ds_labels_list), y=ds_labels_list)
    class_weights_dict = dict(zip(class_labels, class_weights))

    return class_weights_dict


def set_callbacks(train_type=''):
    model_checkpoint = ModelCheckpoint(f'{models_path}{model_name}_{train_type}_best.hdf5',
                                       verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=output_path, histogram_freq=1, write_graph=True, write_images=False)
    csv_logger = CSVLogger(f'{output_path}training_metrics_{train_type}.csv')
    early_stopping = EarlyStopping(patience=stopping_patience, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=lr_patience, min_lr=1e-6)

    return model_checkpoint, tensorboard, csv_logger, early_stopping, reduce_lr


def augment(image, label):
    """Generates augmentations to images."""
    image = tf.image.random_brightness(image, 0.4)
    image = tf.image.random_contrast(image, 0.5, 1.5)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    delta = tf.random.uniform([], math.radians(-360), math.radians(360))
    image = tfa.image.rotate(image, delta)

    return image, label


def apply_performance(ds):
    buffer_size = len(ds)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = (ds.cache().shuffle(buffer_size, reshuffle_each_iteration=False).batch(batch_size).prefetch(AUTOTUNE))

    return ds


def apply_performance_augment(ds):
    buffer_size = len(ds)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = (ds.cache().shuffle(buffer_size, reshuffle_each_iteration=False).batch(batch_size).
          map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))

    return ds


def history_results(fitted_model, run_duration):
    # Evaluate the Model
    loss, acc = model.evaluate(test_ds)

    num_epochs = (fitted_model.epoch[-1]) + 1
    print("\nTraining & Validation Results")
    print(f'Number of training epochs run: {num_epochs}\n')
    print(f'Training Run Time: {run_duration} seconds')
    print(f'Average time per epoch: {run_duration//num_epochs}')

    # Loss
    print(f"\nTraining Loss: {fitted_model.history['loss'][-1]:.3f}")
    print(f"Validation Loss: {fitted_model.history['val_loss'][-1]:.3f}")
    print(f"Evaluation Loss: {loss:.3f}")
    # Accuracy
    print(f"\nTraining Accuracy: {fitted_model.history['sparse_categorical_accuracy'][-1]:.3f}")
    print(f"Validation Accuracy: {fitted_model.history['val_sparse_categorical_accuracy'][-1]:.3f}")
    print(f"Evaluation Accuracy: {acc:.2f}")

    # Save summary to .txt file
    with open(f'{output_path}Training_Summary.txt', 'w') as f:
        print(f'Number of training epochs run: {num_epochs}', file=f)
        print(f'Training Run Time: {run_duration} seconds', file=f)
        print(f'Average time per epoch: {run_duration // num_epochs}', file=f)

        print(f"\nTraining Loss: {fitted_model.history['loss'][-1]:.3f}", file=f)
        print(f"Validation Loss: {fitted_model.history['val_loss'][-1]:.3f}", file=f)
        print(f"Evaluation Loss: {loss:.3f}", file=f)

        print(f"\nTraining Accuracy: {fitted_model.history['sparse_categorical_accuracy'][-1]:.3f}", file=f)
        print(f"Validation Accuracy: {fitted_model.history['val_sparse_categorical_accuracy'][-1]:.3f}", file=f)
        print(f"Evaluation Accuracy: {acc:.2f}", file=f)
    return


def plot_history(fitted_model, train_type=''):
    """Plot the training and validation loss."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(fitted_model.history['loss'], 'g')
    ax[0].plot(fitted_model.history['val_loss'], 'r')
    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper right')
    ax[1].plot(fitted_model.history['sparse_categorical_accuracy'], 'g')
    ax[1].plot(fitted_model.history['val_sparse_categorical_accuracy'], 'r')
    ax[1].set_title('Model accuracy')
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train Accuracy', 'Validation Accuracy'], loc='upper right')
    fig.suptitle(f'{model_name}_{train_type}')  # TODO: Added In Last Version
    fig.savefig(f'{output_path}loss_accuracy_{train_type}.png', bbox_inches='tight', dpi=300)
    return


def gen_confusion_matrix(ds, train_type=''):
    """Calculate True and Predicted Labels. """
    y_true = np.concatenate([y for x, y in ds], axis=0)  # Get true labels
    y_pred = np.argmax(model.predict(ds), axis=1)  # Get predicted labels
    y_pred[np.max(model.predict(ds), axis=1) < 1 / 9] = 8  # Assign predictions < than random guess to negative class

    # Generate classification report
    print(classification_report(y_true, y_pred, target_names=class_names))
    class_report = classification_report(y_true, y_pred, target_names=class_names, digits=3, output_dict=True)
    with open(f'{output_path}Classification_report_{train_type}.csv', 'w') as f:
        for key in class_report.keys():
            f.write("%s,%s\n" % (key, class_report[key]))

    # Generate confusion matrix
    conf_arr = confusion_matrix(y_true, y_pred, labels=class_labels, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_arr, display_labels=class_names)
    disp.plot(xticks_rotation='vertical', values_format='')
    disp.ax_.set_title(f'{model_name}_{train_type}')  # TODO: Added In Last Version
    # Save reports
    disp.figure_.savefig(f'{output_path}Confusion_matrix_{train_type}.png', bbox_inches='tight', dpi=300)
    np.savetxt(f'{output_path}Confusion_matrix_{train_type}.csv', conf_arr, delimiter=",")
    return


def build_model_pretrained():
    """Choose and load a pretrained model from Tensorflow Keras."""
    # Build model for transfer learning
    base_model.trainable = False  # Freeze base_model convolutional layers
    # Resize images and rescale
    inputs = layers.Input(shape=(256, 256, 3), name='source_images')
    x = layers.Resizing(img_size[0], img_size[1], name='resize')(inputs)
    x = preprocess_input(x)
    # Add base then new top
    x = base_model(x,
                   training=False)  # Force inference mode, required for fine-tuning models with BatchNormalization.
    if model_name == 'VGG16':
        x = layers.Flatten()(x)  # if model_choice=4: VGG16
    else:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.4)(x)  # Removes 40% of connections to force redundancy and reduce over training.
    outputs = layers.Dense(units=n_classes, activation='softmax', name='outputs')(x)

    model = Model(inputs, outputs)
    model._name = f"{model_name}_deepweeds"
    return model


def apply_transfer_learning(model, initial_lr=1e-3):
    time_start = time.time()
    # Compile Model for Transfer Learning
    model.compile(optimizer=Adam(learning_rate=initial_lr), loss=LOSS, metrics=METRIC)
    # model.summary()
    print("\nModel is compiled and ready to fit.\n")

    # Set Callbacks for Transfer Learning
    model_checkpoint, tensorboard, csv_logger, early_stopping, reduce_lr = set_callbacks(train_type='TransferLearned')

    # Fit the model
    history = model.fit(train_ds, validation_data=validation_ds, batch_size=batch_size,
                        epochs=tl_epochs, verbose=1, class_weight=class_weights_dict,
                        callbacks=[reduce_lr, early_stopping, tensorboard, model_checkpoint, csv_logger])

    time_end = time.time()
    run_duration = (time_end - time_start)

    # Get Results
    history_results(history, run_duration)
    plot_history(history, train_type='TransferLearned')
    gen_confusion_matrix(test_ds, train_type='TransferLearned')
    return model


def apply_fine_tune(model, initial_lr=1e-5):
    time_start = time.time()

    print('Ensure model is in inference mode if BatchNormalization layers are present')
    # Unfreeze the base model, but ensure that it is still in inference mode.
    model.trainable = True  # This sets all layers excluding BN to train

    # Recompile the model
    model.compile(optimizer=Adam(learning_rate=initial_lr), loss=LOSS, metrics=[METRIC])
    # model.summary()
    # Set Callbacks for fine-tuning
    model_checkpoint, tensorboard, csv_logger, early_stopping, reduce_lr = set_callbacks(train_type='FineTuned')

    # Refit the model
    history = model.fit(train_ds, validation_data=validation_ds, batch_size=batch_size,
                        epochs=ft_epochs, verbose=1, class_weight=class_weights_dict,
                        callbacks=[reduce_lr, early_stopping, tensorboard, model_checkpoint, csv_logger])

    time_end = time.time()
    run_duration = (time_end - time_start)

    # Get Results
    history_results(history, run_duration)
    plot_history(history, train_type='FineTuned')
    gen_confusion_matrix(test_ds, train_type='FineTuned')
    return model


def exit_call():
    """Exits program. Can add exit arg if required."""
    import sys
    return sys.exit()


if __name__ == "__main__":
    # Set Global Parameters
    batch_size = 32  # Set to 32 for balance of performance and memory availability
    tl_epochs = 70  # Maximum number of training epochs for transferring learning.
    ft_epochs = 90  # Maximum number of training epochs for fine-tuning.
    stopping_patience = 15  # number of epochs without improvement before stopping the model training.
    lr_patience = 8  # number of epochs without improvement before changing learning rate.

    # # Set Model Compile variables
    LOSS = 'sparse_categorical_crossentropy'
    METRIC = ['sparse_categorical_accuracy']

    # Load data
    (train_ds, validation_ds, test_ds), info = load_data(data_name="deep_weeds",
                                                         splits=["train[:70%]", "train[70%:85%]", "train[85%:]"])

    # Save example of data
    visualisations = tfds.show_examples(train_ds, info)
    visualisations.savefig(f'{output_path}data_visualisation.png', bbox_inches='tight', dpi=300)

    # Set variables and get data information
    class_names, n_classes, class_labels, img_shape = data_description()

    # Generate tables of dataset information
    nums_df = gen_ds_nums(train_ds, validation_ds, test_ds)

    # Calculate the class weights
    class_weights_dict = calculate_class_weights(train_ds)

    # Apply performance and augmentations
    train_ds = apply_performance_augment(train_ds)
    validation_ds = apply_performance(validation_ds)
    test_ds = apply_performance(test_ds)

    # Start Loop
    # Set Action
    # 1: Load a pretrained model from Tensorflow Keras and train a new head only (Transfer Learning) on new data.
    # 2: Load a saved model and apply transfer learning. This can be used to train further with specified initial_lr.
    # 3: Load a saved model and apply fine-tuning. This can be used to train further with specified initial_lr.
    action = 1

    if action == 1:
        # Select pre_trained model
        # 1: 'ResNet50V2'
        # 2: 'InceptionV3'
        # 3: 'Xception'
        # 4: 'VGG16'
        # 5: 'MobileNetV2'
        # 6: 'DenseNet121'
        # 7: 'EfficientNetV2B1'
        selection = 4

    else:
        # Place file path as string to model file (.hdf5) to load.
        model_file_path_list = ['']

        if action == 2:
            initial_lr = 1e-4  # Or select smallest lr from previous training run.
        else:
            initial_lr = 1e-5  # Learning rate for fine-tuning.

    # Set number of runs
    runs = 1
    if action == 1:
        for i in range(runs):
            # Load pretrained model from keras applications and apply transfer learning.
            (model_name, base_model, preprocess_input), img_size = ModelDict().select_model()
            folder_name, save_path, models_path, output_path = create_folders(model_name)
            save_df(nums_df)
            model = build_model_pretrained()
            model = apply_transfer_learning(model, initial_lr=1e-3)

    elif action == 2:
        for existing_model in model_file_path_list:
            # model_file_path = model_file_path_list[]
            model_file_path = existing_model
            print(model_file_path)
            # Load a model from local and apply transfer learning.
            model = load_model_local(model_file_path)
            model_name, models_path, output_path = get_folders(model_file_path)
            save_df(nums_df)
            model = apply_transfer_learning(model, initial_lr=initial_lr)

    elif action == 3:
        for existing_model in model_file_path_list:
            # model_file_path = model_file_path_list[]
            model_file_path = existing_model
            print(model_file_path)
            # Load a model from local and apply fine tuning.
            model = load_model_local(model_file_path)
            model_name, models_path, output_path = get_folders(model_file_path)
            save_df(nums_df)
            model = apply_fine_tune(model, initial_lr=initial_lr)

    else:
        print('No valid action was selected.')

    # Clear session from memory
    tf.keras.backend.clear_session()
