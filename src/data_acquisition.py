"""Data acquisition: export a subset of PlantVillage (tomato leaves) to data/train and data/test."""
import os, random
from pathlib import Path
from PIL import Image

CLASSES = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold"]
PLANT_VILLAGE_DIR = "Plant_leave_diseases_dataset_without_augmentation"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _cached_plant_village_root():
    """Find TFDS's extracted PlantVillage files without relying on TensorFlow globbing."""
    cache_dirs = []
    if os.environ.get("TFDS_DATA_DIR"):
        cache_dirs.append(Path(os.environ["TFDS_DATA_DIR"]))
    cache_dirs.extend((Path.home() / "tensorflow_datasets", Path("data/tfds")))
    for cache_dir in cache_dirs:
        extracted_dir = cache_dir / "downloads" / "extracted"
        if extracted_dir.is_dir():
            matches = sorted(extracted_dir.glob(f"*/{PLANT_VILLAGE_DIR}"))
            if matches:
                return matches[0]
    return None


def _export_cached_images(source_dir, per_class, test_frac, out):
    rng = random.Random(42)
    counts = {c: 0 for c in CLASSES}
    for c in CLASSES:
        class_dir = source_dir / c
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        selected = rng.sample(images, min(per_class, len(images)))
        for index, path in enumerate(selected):
            split = "test" if rng.random() < test_frac else "train"
            destination = Path(out) / split / c / f"{c}_{index}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                image.convert("RGB").resize((256, 256)).save(destination)
            counts[c] += 1
    print("Source:", source_dir)
    print("Saved:", counts)

def acquire(per_class=400, test_frac=0.2, out="data"):
    source_dir = _cached_plant_village_root()
    if source_dir is not None:
        _export_cached_images(source_dir, per_class, test_frac, out)
        return

    import tensorflow_datasets as tfds
    try:
        ds, info = tfds.load("plant_village", split="train", with_info=True, as_supervised=True)
    except AssertionError:
        # Some Windows TFDS builds fail to match the mixed-case JPG files while
        # generating TFRecords. The archive is already extracted at that point.
        source_dir = _cached_plant_village_root()
        if source_dir is None:
            raise
        _export_cached_images(source_dir, per_class, test_frac, out)
        return
    names = info.features["label"].names
    ids = {names.index(c): c for c in CLASSES}
    counts = {c: 0 for c in CLASSES}
    for img, label in tfds.as_numpy(ds):
        label = int(label)
        if label not in ids:
            continue
        c = ids[label]
        if counts[c] >= per_class:
            if all(v >= per_class for v in counts.values()):
                break
            continue
        split = "test" if random.random() < test_frac else "train"
        d = os.path.join(out, split, c)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(img).resize((256, 256)).save(os.path.join(d, f"{c}_{counts[c]}.jpg"))
        counts[c] += 1
    print("Saved:", counts)

if __name__ == "__main__":
    acquire()
