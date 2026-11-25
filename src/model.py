"""Model creation, training, testing and retraining."""
import json, os, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from src.preprocessing import get_datasets, augmentation, merge_uploads, IMG_SIZE

MODEL_PATH = "models/leaf_model.h5"
CLASSES_PATH = "models/class_names.json"
METRICS_PATH = "models/metrics.json"
EARLY_BLIGHT_CLASS = "Tomato___Early_blight"
EARLY_BLIGHT_THRESHOLD = 0.30

def classify_probabilities(probabilities, names):
    predicted = np.argmax(probabilities, axis=1)
    if EARLY_BLIGHT_CLASS in names:
        early_index = names.index(EARLY_BLIGHT_CLASS)
        # This validation-tuned operating point prioritizes early-blight recall over argmax.
        predicted[probabilities[:, early_index] >= EARLY_BLIGHT_THRESHOLD] = early_index
    return predicted

def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    inp = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation()(inp)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def evaluate(model, test_ds, names):
    y_true, y_pred = [], []
    for x, y in test_ds:
        y_true += y.numpy().tolist()
        y_pred += classify_probabilities(model.predict(x, verbose=0), names).tolist()
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    loss = float(model.evaluate(test_ds, verbose=0)[0])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
        "recall": float(r), "f1": float(f), "loss": loss,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=names, output_dict=True, zero_division=0),
        "classes": names,
        "decision_policy": {"early_blight_threshold": EARLY_BLIGHT_THRESHOLD},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

def train(epochs=8, fine_tune_epochs=3, train_dir="data/train", test_dir="data/test"):
    train_ds, val_ds, test_ds, names = get_datasets(train_dir, test_dir)
    model = build_model(len(names))
    cb = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
    if fine_tune_epochs:
        base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
        base.trainable = True
        for l in base.layers[:-30]:
            l.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=cb)
    metrics = evaluate(model, test_ds, names)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    json.dump(names, open(CLASSES_PATH, "w"))
    json.dump(metrics, open(METRICS_PATH, "w"), indent=2)
    return model, metrics, hist

def retrain(**kw):
    """Merge newly uploaded labelled data into data/train, then train again."""
    merged = merge_uploads()
    model, metrics, _ = train(**kw)
    metrics["new_images_merged"] = merged
    json.dump(metrics, open(METRICS_PATH, "w"), indent=2)
    return metrics

if __name__ == "__main__":
    _, m, _ = train()
    print({k: m[k] for k in ("accuracy", "precision", "recall", "f1", "loss")})
