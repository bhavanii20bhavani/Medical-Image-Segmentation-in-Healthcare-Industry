import cv2
import numpy as np


def OACF(image, scale_factor, min_neighbors, min_size):
    resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Compute gradient magnitude and angle
    gradients_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag, gradient_angle = cv2.cartToPolar(gradients_x, gradients_y, angleInDegrees=True)

    # Normalize gradient magnitude
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert gradient magnitude to 8-bit
    gradient_mag = gradient_mag.astype(np.uint8)

    # Compute histogram of gradient orientations
    gradient_hist = cv2.calcHist([gradient_angle], [0], None, [8], [0, 360])
    gradient_hist = cv2.normalize(gradient_hist, gradient_hist).flatten()

    # Extract color channels
    b, g, r = cv2.split(resized_image)

    # Concatenate features
    features = cv2.vconcat([gray, gradient_mag, b, g, r])
    features = cv2.resize(features, (min_size, min_size))  # Adjust size

    # Here, you might include more sophisticated use of min_neighbors if you were doing detection.
    # As an example, let's simulate neighbor influence by further blurring the image if min_neighbors is high.
    if min_neighbors > 10:
        features = cv2.GaussianBlur(features, (min_neighbors, min_neighbors), 0)
    features = cv2.resize(features, (image.shape))
    return features


def Model_OACF(images, sol=None):
    if sol is None:
        sol = [0.5, 3, 12]
    scale = sol[0]
    neighbors = sol[1]
    size = sol[2]
    acf_features = []
    for image in images:
        features = OACF(image, scale, neighbors, int(size))
        acf_features.append(features)
    acf_features = np.array(acf_features)
    return acf_features

