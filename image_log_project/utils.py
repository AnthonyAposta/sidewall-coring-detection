import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import ceil


def plot_images(
    images: list,
    contours_list: np.array = None,
    pred_centroids: list = None,
    true_centroids: list = None,
    n=16,
):
    images = images[:n]

    n_images = len(images)
    n_columns = 4
    n_rows = ceil(n_images / n_columns)

    x_size = 12
    y_size = int(n_rows * x_size / n_columns)

    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(x_size, y_size), sharex=True, sharey=True
    )

    plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=0.05, hspace=0.05)

    ax = axes.flatten()

    for axs in ax[n_images:]:
        axs.remove()

    for i, img in enumerate(images):
        ax[i].set_axis_off()

        image = img.copy()
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if contours_list is not None:
            img_contours = contours_list[i]
            cv2.drawContours(
                image, img_contours, -1, (0, 255, 0), 2
            )  # -1 means draw all contours, (0, 255, 0) is the color, 2 is the thickness

        # red
        if pred_centroids is not None:
            for cntr in pred_centroids[i]:
                cv2.circle(image, cntr, radius=5, color=(255, 0, 0, 100), thickness=2)

        # green
        if true_centroids is not None:
            for cntr in true_centroids[i]:
                cv2.circle(image, cntr, radius=5, color=(0, 255, 0, 100), thickness=2)

        ax[i].imshow(image)

    plt.show()
