import time
from datetime import timedelta
from shutil import copyfile
import numpy as np
import cv2
import os
from tqdm import tqdm
import settings



# This function smooth our transformations
def smoothing(array, r=10):
    window = 2 * r + 1
    smoothed_array = np.copy(array)
    # Smoothing dx, dy, da
    for i in range(3):
        smoothed_array[:, i] = np.convolve(smoothed_array[:, i], np.ones(window) / window, mode='same')
    return smoothed_array


# This function rescale our image to remove the black borders
def removingBlackBorders(frame):
    rescale_factor = 1.02
    T = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), 0, rescale_factor)
    frame = cv2.warpAffine(frame, T, (frame.shape[1], frame.shape[0]))
    return frame


def Stabilize_Video(input_video, output_video):
    # Parameters
    maxCorners = 200
    qualityLevel = 0.01
    minDistance = 50
    blockSize = 5

    # Read input
    cap = cv2.VideoCapture(input_video)

    # Get video parameters
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # output video
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # Read first frame and convert frame to grayscale
    _, prev_frame = cap.read()
    out.write(removingBlackBorders(prev_frame))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Set up array
    transformations = np.zeros((frameCount - 1, 3), np.float32)

    for i in tqdm(range(frameCount - 1)):
        # Detect corners
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=maxCorners, qualityLevel=qualityLevel,
                                           minDistance=minDistance, blockSize=blockSize)

        # Read next frame and convert to grayscale
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Find frame movement
        curr_pts, valid, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # filter the points
        idx = np.where(valid == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        """ # RANSAC - Removing Outliers (Didnt improve results)
        # Use RANSAC to remove outliers
        M, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
        for i in range(len(mask) - 1, -1, -1):
            if mask[i, 0] == 0:
                prev_pts = np.delete(prev_pts, i, axis=0)
                curr_pts = np.delete(curr_pts, i, axis=0)
        """
        # Get transformation matrix
        transformation_mat = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        # Extract traslation
        dx = transformation_mat[0, 2]
        dy = transformation_mat[1, 2]
        da = np.arctan2(transformation_mat[1, 0], transformation_mat[0, 0])
        # Store transformation
        transformations[i] = [dx, dy, da]

        # Next frame
        prev_gray = curr_gray

    # Sum up to cumulative transformations
    cumulative_transformations = np.cumsum(transformations, axis=0)
    # Smooth the transformations
    diff = smoothing(cumulative_transformations) - cumulative_transformations
    smoothed_transformations = transformations + diff

    # warp frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset reading from video
    for i in range(frameCount - 1):
        success, frame = cap.read()  # next frame
        if not success:
            break

        # get dx dy da
        dx = smoothed_transformations[i, 0]
        dy = smoothed_transformations[i, 1]
        da = smoothed_transformations[i, 2]

        # build transformation_mat
        transformation_mat = np.zeros((2, 3), np.float32)
        transformation_mat[0, 0] = np.cos(da)
        transformation_mat[0, 1] = -np.sin(da)
        transformation_mat[1, 0] = np.sin(da)
        transformation_mat[1, 1] = np.cos(da)
        transformation_mat[0, 2] = dx
        transformation_mat[1, 2] = dy

        # warp the frame
        frame_stabilized = cv2.warpAffine(frame, transformation_mat, (w, h))
        # remove black borders
        frame_stabilized = removingBlackBorders(frame_stabilized)

        # write to file
        out.write(frame_stabilized)

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()
    return smoothed_transformations


def Stabilize_Video_Main():
    start_time = time.time()
    print('### STARTING Stabilization ###')

    input_video = settings.input_name
    output_video = settings.stabilized_name

    parameter_count = 3  # dx dy da
    iterations = settings.Stabilization_iterations
    cap = cv2.VideoCapture(input_video)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    All_transformations = np.zeros((iterations, frameCount, parameter_count), np.float32)

    basename = os.path.basename(input_video)
    copyfile(input_video, settings.tempdir+basename+'_0.avi')
    i = 0

    while i < iterations - 1:
        print('Stabilize iteration {0}/{1}.'.format(i, iterations))
        time.sleep(0.01)  # printing without overrunning the bar
        inputname =settings.tempdir+basename+'_'+str(i)+'.avi'
        outputname = settings.tempdir+basename+'_'+str(i+1)+'.avi'

        transformation = Stabilize_Video(inputname, outputname)

        # Saving the transformation for later use
        pad = frameCount - len(transformation)
        testpad = np.pad(transformation, [(0, pad), (0, 0)], mode='constant')
        All_transformations[i, :, :] = testpad
        i += 1
        os.remove(inputname)

    print('Stabilize iteration {0}/{1}.'.format(i, iterations))
    inputname = settings.tempdir+basename+'_'+str(i)+'.avi'
    transformation = Stabilize_Video(inputname, output_video)
    os.remove(inputname)

    # Saving the transformation for later use
    pad = frameCount - len(transformation)
    All_transformations[i, :, :] = np.pad(transformation, [(0, pad), (0, 0)], mode='constant')
    np.append(All_transformations, transformation)

    print('### FINISHED Stabilization. Run time: {:0>8} ###'.format(str(timedelta(seconds=(int(time.time() - start_time))))))
    print('#######################################')
    return All_transformations
