from datetime import timedelta
import time
import cv2
import numpy as np
from tqdm import tqdm
import auxiliary_functions as AF
import settings


def backgroundSubtraction():
    stabilized_video = settings.stabilized_name
    binary_video = settings.binary_video
    extracted_name = settings.extracted_name

    # Getting Video parameters
    cap_stabilized = cv2.VideoCapture(stabilized_video)
    fourcc = cap_stabilized.get(6)
    fps = cap_stabilized.get(5)
    frameSize = (int(cap_stabilized.get(3)), int(cap_stabilized.get(4)))
    numFrames = int(cap_stabilized.get(7))
    # Setting output file
    out_binary = cv2.VideoWriter(binary_video, int(fourcc), fps, frameSize, 0)
    out_extract = cv2.VideoWriter(extracted_name, int(fourcc), fps, frameSize)

    # first Background Subtractor
    mask = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=100, detectShadows=False)

    for fr in tqdm(range(numFrames)):
        ret, frame = cap_stabilized.read()
        frame_color = frame
        if ret:
            # Changing to HSV color for better accuracy
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame = hsv[:, :, 1]

            # Declaring kernels
            kernel8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_long = np.ones((2,10))

            # Applying mask
            frame1 = mask.apply(frame)
            frame1 = frame1.astype(np.uint8)

            # Filtering black/white by threshold
            th, frame1 = cv2.threshold(frame1, 150, 255, cv2.THRESH_BINARY)

            # Morphology: Gradient and closing to complete missing parts of object
            frame1 = cv2.morphologyEx(frame1, cv2.MORPH_GRADIENT, kernel4, iterations=5)
            frame1 = cv2.morphologyEx(frame1, cv2.MORPH_CLOSE, kernel2, iterations=1)

            # Connected Components - Clear all noises that are not the object
            nb_components, output, sizes, centers, max_label, max_size = AF.connectedComponentsWithSizesAndMax(frame1)
            frame1 = np.zeros(output.shape)
            if nb_components > 1:
                frame1[output == max_label] = 255
            frame1 = frame1.astype(np.uint8)

            # Filling the object with '1'
            frame1 = AF.fill(frame1)

            # completing and filling more parts of the object
            frame1 = cv2.morphologyEx(frame1, cv2.MORPH_CLOSE, kernel8, iterations=5)
            frame1 = frame1.astype(np.uint8)
            frame1 = AF.fill(frame1)

            # Morphology: opening with different kernels to delete parts that are
            # connected to the object but not part of him, then closing to complete holes
            frame1 = cv2.morphologyEx(frame1, cv2.MORPH_OPEN, kernel8, iterations=1)
            frame1 = cv2.morphologyEx(frame1, cv2.MORPH_OPEN, kernel_long, iterations=3)
            frame1 = cv2.morphologyEx(frame1, cv2.MORPH_OPEN, kernel_long.T, iterations=3)
            frame1 = cv2.morphologyEx(frame1, cv2.MORPH_CLOSE, kernel8, iterations=5)

            # dilating a little to get a "spare" mask around our target and not to miss any part of it
            frame1 = cv2.dilate(frame1, kernel8, iterations=1)

            # Preparing the 'extracted' video
            final_frame = np.ones_like(frame_color) * 255
            final_frame2 = final_frame.copy()
            final_frame2[frame1 == 255] = frame_color[frame1 == 255]

            frame1 = frame1.astype(np.uint8)
            frame2 = final_frame2.astype(np.uint8)

            # writing to output video
            out_binary.write(frame1)
            out_extract.write(frame2)
        else:
            break

    out_binary.release()
    out_extract.release()
    cap_stabilized.release()



def backgroundSubtraction_Main():
    start_time = time.time()

    print('### STARTING Background Subtraction ###')
    backgroundSubtraction()
    print('### FINISHED Background Subtraction. Run time: {:0>8} ###'.format(str(timedelta(seconds=(int(time.time() - start_time))))))
    print('#######################################')
