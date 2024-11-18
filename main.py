import os
import cv2 # type: ignore
import numpy as np # type: ignore

# Helper function to read image and convert to binary (assuming binary image for labeling)
def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to open image file {image_path}")
        return None
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary_img

# Task 1: Component Labeling Algorithm (4-connectivity)
def component_labeling_4_connected(binary_img):
    h, w = binary_img.shape
    labeled_img = np.zeros((h, w), dtype=np.int32)
    label = 0

    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 255 and labeled_img[i, j] == 0:
                label += 1
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if labeled_img[x, y] == 0:
                        labeled_img[x, y] = label
                        # Check 4-connectivity neighbors
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and binary_img[nx, ny] == 255 and labeled_img[nx, ny] == 0:
                                stack.append((nx, ny))
    return labeled_img

# Task 2: Component Labeling Algorithm (8-connectivity)
def component_labeling_8_connected(binary_img):
    h, w = binary_img.shape
    labeled_img = np.zeros((h, w), dtype=np.int32)
    label = 0

    for i in range(h):
        for j in range(w):
            if binary_img[i, j] == 255 and labeled_img[i, j] == 0:
                label += 1
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if labeled_img[x, y] == 0:
                        labeled_img[x, y] = label
                        # Check 8-connectivity neighbors
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and binary_img[nx, ny] == 255 and labeled_img[nx, ny] == 0:
                                stack.append((nx, ny))
    return labeled_img

# Task 3: Component Labeling with Intensity Range
def component_labeling_intensity_range(img, min_val, max_val):
    h, w = img.shape
    labeled_img = np.zeros((h, w), dtype=np.int32)
    label = 0

    def is_within_range(val):
        return min_val <= val <= max_val

    for i in range(h):
        for j in range(w):
            if is_within_range(img[i, j]) and labeled_img[i, j] == 0:
                label += 1
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if labeled_img[x, y] == 0:
                        labeled_img[x, y] = label
                        # Check 8-connectivity neighbors
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and is_within_range(img[nx, ny]) and labeled_img[nx, ny] == 0:
                                stack.append((nx, ny))
    return labeled_img

# Task 4: Size Filter Algorithm
def size_filter(labeled_img, min_size, max_size):
    h, w = labeled_img.shape
    unique_labels, counts = np.unique(labeled_img, return_counts=True)
    size_filtered_img = np.zeros((h, w), dtype=np.uint8)

    for label, count in zip(unique_labels, counts):
        if min_size <= count <= max_size:
            size_filtered_img[labeled_img == label] = 255
    return size_filtered_img

# Example usage:
if __name__ == "__main__":
    # Use absolute path to ensure the file can be found
    image_path = r'D:\OneDrive\Desktop\imageProject\imageProject\images.jpeg'
    binary_img = read_image(image_path)

    if binary_img is not None:
        # Task 1
        labeled_img_4 = component_labeling_4_connected(binary_img)
        cv2.imwrite('labeled_4_connected.png', labeled_img_4.astype(np.uint8) * 10)

        # Task 2
        labeled_img_8 = component_labeling_8_connected(binary_img)
        cv2.imwrite('labeled_8_connected.png', labeled_img_8.astype(np.uint8) * 10)

        # Task 3
        min_val, max_val = 50, 200
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        labeled_img_intensity = component_labeling_intensity_range(img, min_val, max_val)
        cv2.imwrite('labeled_intensity_range.png', labeled_img_intensity.astype(np.uint8) * 10)

        # Task 4
        min_size, max_size = 500, 2000
        size_filtered_img = size_filter(labeled_img_8, min_size, max_size)
        cv2.imwrite('size_filtered.png', size_filtered_img)
