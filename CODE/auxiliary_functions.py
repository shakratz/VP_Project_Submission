import cv2
import numpy as np
import time
from tqdm import tqdm
import settings


def fill(frame):
    # Copy the thresholded image.
    frame_floodfill = frame.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = frame.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(frame_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    frame_floodfill_inv = cv2.bitwise_not(frame_floodfill)

    # Combine the two images to get the foreground.
    frame_filled = frame | frame_floodfill_inv
    return frame_filled


def edge_noise_cleaning(frame, iterations):
    # define kernel to clean edges-noises
    kernel = np.zeros((4, 10))
    kernel[:, 1:9] = 1
    kernel = kernel.astype(np.uint8)

    # cleaning noises that produced by unstable edges
    frame = cv2.erode(frame, kernel, iterations=iterations)  # clean horizontal noises
    frame = cv2.erode(frame, kernel.T, iterations=iterations)  # clean vertical noises
    return frame


def connectedComponentsWithSizesAndMax(frame):
    nb_components, output, stats, centers = cv2.connectedComponentsWithStats(frame, connectivity=8)
    sizes = stats[:, -1]  # sizes of white parts
    second_max = 1
    max_label = 1
    if len(sizes) > 1:
        max_size = sizes[1]
    else:
        max_size = 0

    # we check what the size of the bigest white part
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            second_max = max_label
            max_label = i
            max_size = sizes[i]

    return nb_components, output, sizes, centers, max_label, max_size


def CreatingUnstabilizedAlpha(All_transformations, unstabilized_alpha_name):
    alpha_name = settings.alpha_name
    # Read input
    cap = cv2.VideoCapture(alpha_name)

    # Get video parameters
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    iterations = settings.Stabilization_iterations  # same as stabilization block

    # output video
    output_name = unstabilized_alpha_name
    out = cv2.VideoWriter(output_name, fourcc, fps, (w, h))

    unstable_video = np.zeros((h, w, 3, frameCount), np.float32)
    for j in range(iterations):
        time.sleep(0.01)  # printing without overrunning the bar
        print('Building Unstable alpha, iteration :{0}/{1}.'.format(j + 1, iterations))
        time.sleep(0.01)    # printing without overrunning the bar
        for i in tqdm(range(frameCount)):
            if j == 0:
                success, frame = cap.read()  # next frame
                if not success:
                    break
            else:
                frame = unstable_video[:, :, :, i]

            transform = All_transformations[iterations - 1 - j, i, :]
            transform = transform * (-1)  # reversing the transform
            # get dx dy da
            dx = transform[0]
            dy = transform[1]
            da = transform[2]

            # build transformation_mat
            transformation_mat = np.zeros((2, 3), np.float32)
            transformation_mat[0, 0] = np.cos(da)
            transformation_mat[0, 1] = -np.sin(da)
            transformation_mat[1, 0] = np.sin(da)
            transformation_mat[1, 1] = np.cos(da)
            transformation_mat[0, 2] = dx
            transformation_mat[1, 2] = dy

            # warp the frame
            frame_unstabilized = cv2.warpAffine(frame, transformation_mat, (w, h))

            # write to file
            unstable_video[:, :, :, i] = frame_unstabilized
            if j == iterations - 1:
                frame_unstabilized = frame_unstabilized.astype(np.uint8)
                out.write(frame_unstabilized)

    cap.release()
    out.release()

