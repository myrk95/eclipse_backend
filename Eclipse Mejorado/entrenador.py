import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("TensorFlow:", tf.__version__)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "./"
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

N_MALIGNANT = 4521
N_BENIGN = 4521

TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10

assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 1e-6

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40

OUT_DIR = "./modelos_guardados_isic2019"
os.makedirs(OUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUT_DIR, "isic2019_mobilenetv2_best.keras")

print("CSV_PATH:", CSV_PATH)
print("IMAGES_DIR:", IMAGES_DIR)
print("BEST_MODEL_PATH:", BEST_MODEL_PATH)

df = pd.read_csv(CSV_PATH)
df = df[df["benign_malignant"].isin(["benign", "malignant"])].copy()

df["image_stem"] = df["image_name"].astype(str).str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)

df["filepath"] = df["image_stem"].apply(lambda s: os.path.join(IMAGES_DIR, f"{s}.jpg"))

df["label"] = (df["benign_malignant"] == "malignant").astype(int)

exists_mask = df["filepath"].apply(os.path.exists)
df = df[exists_mask].copy()

print("Total filas válidas:", len(df))
print(df[["benign_malignant","label"]].value_counts())

df_m = df[df["label"] == 1].sample(n=N_MALIGNANT, random_state=SEED)
df_b = df[df["label"] == 0].sample(n=N_BENIGN, random_state=SEED)
df_bal = pd.concat([df_m, df_b]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print("Distribución en dataset:")
print("  Benignos:", (df_bal["label"]==0).sum(), f"({(df_bal['label']==0).mean()*100:.1f}%)")
print("  Malignos:", (df_bal["label"]==1).sum(), f"({(df_bal['label']==1).mean()*100:.1f}%)")

df_train, df_tmp = train_test_split(
    df_bal, test_size=(1.0-TRAIN_SIZE), random_state=SEED, stratify=df_bal["label"]
)

val_fraction_of_tmp = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
df_val, df_test = train_test_split(
    df_tmp, test_size=(1.0-val_fraction_of_tmp), random_state=SEED, stratify=df_tmp["label"]
)

print("DIVISIÓN DE DATOS:")
print(f"  Train: {len(df_train)} muestras ({len(df_train)/len(df_bal)*100:.1f}%)")
print(f"  Validación: {len(df_val)} muestras ({len(df_val)/len(df_bal)*100:.1f}%)")
print(f"  Test: {len(df_test)} muestras ({len(df_test)/len(df_bal)*100:.1f}%)")
print(f"  Total: {len(df_bal)} muestras")

print("\nDistribución por clase (Train):")
print(df_train["label"].value_counts().rename({0:"Benignos",1:"Malignos"}))

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_dataframe(
    df_train,
    x_col="filepath",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_dataframe(
    df_val,
    x_col="filepath",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_gen = val_datagen.flow_from_dataframe(
    df_test,
    x_col="filepath",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False
)

base = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()

callbacks = [
    ModelCheckpoint(BEST_MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

y_true = df_test["label"].values
y_prob = best_model.predict(test_gen, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

test_auc = roc_auc_score(y_true, y_prob)
print("Test AUC:", round(test_auc, 4))
print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))