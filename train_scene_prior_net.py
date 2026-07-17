"""
train_scene_prior_net.py - 原始版本（无数据增强）
"""
import tensorflow as tf
import numpy as np
import glob
import os
import argparse
from sklearn.model_selection import train_test_split
from scene_prior_net import build_scene_prior_net

def load_data(data_dir, target_size=(256,256)):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    X, Y = [], []
    for f in files:
        data = np.load(f)
        img = data['image'].astype(np.float32) / 255.0
        mask = data['mask'].astype(np.float32)
        X.append(img[..., np.newaxis])
        Y.append(mask[..., np.newaxis])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_save', default='scene_prior_net.h5')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--input_size', type=str, default='256,256')
    args = parser.parse_args()
    input_size = tuple(map(int, args.input_size.split(',')))

    X, Y = load_data(args.data_dir, input_size)
    print(f"Loaded {len(X)} samples, X shape {X.shape}, Y shape {Y.shape}")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_scene_prior_net(input_shape=(input_size[0], input_size[1], 1), num_filters=32)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.model_save, monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]
    model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks)
    model.save(args.model_save)
    print(f"Model saved to {args.model_save}")

if __name__ == "__main__":
    main()