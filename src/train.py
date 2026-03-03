import tensorflow as tf
from src.data_preprocessing import create_data_generators
from src.model import build_model
import os

def main():
    # Use the split dataset
    train_gen, val_gen, test_gen = create_data_generators()
    
    model = build_model()  # Fixed: underscore, not space
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,  # Fixed: batch_size (underscore)
        epochs=10   # You can increase later, start with 10 for testing
    )
    
    # Create models folder if not exists
    os.makedirs('models', exist_ok=True)
    model.save('models/crack_detection_mobilenet_cbam.h5')
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()