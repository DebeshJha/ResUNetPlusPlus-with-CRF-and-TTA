
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from sklearn.utils import shuffle

from unet import Unet
from resunet import ResUnet
from m_resunet import ResUnetPlusPlus
from metrics import *
from tf_data import *
from sgdr import *

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    ## Path
    file_path = "files/"

    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass

    train_path = "new_data/train/"
    valid_path = "new_data/valid/"

    ## Training
    train_image_paths = sorted(glob(os.path.join(train_path, "image", "*.jpg")))
    train_mask_paths = sorted(glob(os.path.join(train_path, "mask", "*.jpg")))

    ## Shuffling
    train_image_paths, train_mask_paths = shuffling(train_image_paths, train_mask_paths)

    ## Validation
    valid_image_paths = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_mask_paths = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))

    ## Parameters
    image_size = 256
    batch_size = 16
    lr = 1e-5
    epochs = 300
    model_path = "files/resunetplusplus.h5"

    train_dataset = tf_dataset(train_image_paths, train_mask_paths)
    valid_dataset = tf_dataset(valid_image_paths, valid_mask_paths)

    try:
        arch = ResUnetPlusPlus(input_size=image_size)
        model = arch.build_model()
        model = tf.distribute.MirroredStrategy(model, 4, cpu_merge=False)
        print("Training using multiple GPUs..")
    except:
        arch = ResUnetPlusPlus(input_size=image_size)
        model = arch.build_model()
        print("Training using single GPU or CPU..")

    optimizer = Nadam(learning_rate=lr)
    metrics = [dice_coef, MeanIoU(num_classes=2), Recall(), Precision()]

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    model.summary()
    schedule = SGDRScheduler(min_lr=1e-6,
                             max_lr=1e-2,
                             steps_per_epoch=np.ceil(epochs/batch_size),
                             lr_decay=0.9,
                             cycle_length=5,
                             mult_factor=1.5)

    callbacks = [
        ModelCheckpoint(model_path),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
        schedule
    ]

    train_steps = (len(train_image_paths)//batch_size)
    valid_steps = (len(valid_image_paths)//batch_size)

    if len(train_image_paths) % batch_size != 0:
        train_steps += 1

    if len(valid_image_paths) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
            epochs=epochs,
            validation_data=valid_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks,
            shuffle=False)

