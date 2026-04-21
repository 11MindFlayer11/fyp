import numpy as np
import cv2

# squares (NOT inner corners)
rows = 7   # squares
cols = 10

square_size = 100  # pixels per square

img = np.zeros((rows * square_size, cols * square_size), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            cv2.rectangle(
                img,
                (j * square_size, i * square_size),
                ((j + 1) * square_size, (i + 1) * square_size),
                255,
                -1
            )

cv2.imwrite("calibration/checkerboard.png", img)