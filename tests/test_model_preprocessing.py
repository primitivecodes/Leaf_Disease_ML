import unittest

import numpy as np
import tensorflow as tf

from src.model import (
    EARLY_BLIGHT_CLASS,
    EARLY_BLIGHT_THRESHOLD,
    classify_probabilities,
)
from src.preprocessing import IMG_SIZE, preprocess_bytes


class ModelPreprocessingTests(unittest.TestCase):
    def test_preprocessing_matches_tensorflow_training_resize(self):
        image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape((24, 32, 3))
        encoded = tf.io.encode_jpeg(image).numpy()
        decoded = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        expected = tf.image.resize(decoded, IMG_SIZE, method="bilinear").numpy()

        actual = preprocess_bytes(encoded)

        self.assertEqual(actual.shape, (1, *IMG_SIZE, 3))
        np.testing.assert_array_equal(actual[0], expected)

    def test_early_blight_threshold_overrides_highest_probability(self):
        names = [EARLY_BLIGHT_CLASS, "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___healthy"]
        probabilities = np.array([[EARLY_BLIGHT_THRESHOLD, 0.50, 0.10, 0.10]])

        predicted = classify_probabilities(probabilities, names)

        self.assertEqual(predicted.tolist(), [0])

    def test_argmax_is_used_below_early_blight_threshold(self):
        names = [EARLY_BLIGHT_CLASS, "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___healthy"]
        probabilities = np.array([[EARLY_BLIGHT_THRESHOLD - 0.001, 0.50, 0.10, 0.10]])

        predicted = classify_probabilities(probabilities, names)

        self.assertEqual(predicted.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
