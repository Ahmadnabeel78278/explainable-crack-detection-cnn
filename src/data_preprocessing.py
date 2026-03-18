from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(
    train_dir='data_splitted/train',
    val_dir='data_splitted/val',
    test_dir='data_splitted/test',
    img_size=(224,224),
    batch_size=32
):
<<<<<<< HEAD
    # Training data generator with augmentation
=======

>>>>>>> b48b3055 (Complete project with all source code and updates)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
<<<<<<< HEAD
    # Validation/test generator (only rescaling)
=======
>>>>>>> b48b3055 (Complete project with all source code and updates)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator