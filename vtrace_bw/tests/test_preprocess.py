import unittest
import numpy as np
from vtrace.preprocess import rectify_perspective, preprocess_image


class PreprocessTests(unittest.TestCase):
    def test_rectify_empty_or_small(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        out = rectify_perspective(img)
        self.assertEqual(out.shape, (10, 10, 4))

    def test_rectify_preserves_alpha_border(self):
        # Create an image with black rectangle and transparent background
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[20:80, 20:80, :3] = 0  # Black box
        img[20:80, 20:80, 3] = 255 # Opaque

        out = rectify_perspective(img)
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[2], 4)
        # Background borders must remain transparent (alpha == 0)
        self.assertEqual(out[0, 0, 3], 0)

    def test_preprocess_image_grayscale(self):
        gray = np.full((50, 50), 128, dtype=np.uint8)
        out = preprocess_image(gray, rectify=False, remove_bg=False)
        self.assertEqual(out.shape, (50, 50, 4))
        self.assertEqual(out.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
