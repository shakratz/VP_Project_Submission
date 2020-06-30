import time
from datetime import timedelta
import cv2
from tqdm import tqdm
import settings

def tracking(ManualTracking):
    matted_video = settings.matted_name
    output_name = settings.output_name

    # unfortuntly only from the ~10th frame we have the full object
    startFromFrameNum = settings.startFromFrameNum

    # Getting Video parameters
    cap = cv2.VideoCapture(matted_video)
    fourcc = cap.get(6)
    fps = cap.get(5)
    frameSize = (int(cap.get(3)), int(cap.get(4)))
    numFrames = int(cap.get(7))

    # Setting output file
    out = cv2.VideoWriter(output_name, int(fourcc), fps, frameSize)

    # load first frame
    for j in range(startFromFrameNum):
        ret, first_frame = cap.read()

    # Get the initial object position from user/hardcoded
    if ManualTracking:
        print('Please Select the object, then press SPACE')
        rect = cv2.selectROI('select', first_frame)
        cv2.destroyWindow('select')
    else:
        # Values for a decent result:
        rect = (72, 227, 236, 716)

    # adding the cv boosting tracker
    tracker = cv2.TrackerBoosting_create()
    ok = tracker.init(first_frame, rect)

    # Looping over each frame
    for fr in tqdm(range(startFromFrameNum, numFrames)):
        # LOAD NEW FRAME
        ret, frame = cap.read()

        # updating our rectangle position
        ok, rect = tracker.update(frame)

        # plotting the new rectangle
        p1 = (int(rect[0]), int(rect[1]))
        p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        out.write(frame)

    cap.release()
    out.release()

def Tracking_Main(ManualTracking):
    start_time = time.time()

    print('### STARTING Tracking ###')
    tracking(ManualTracking)
    print('### FINISHED Tracking. Run time: {:0>8}'.format(str(timedelta(seconds=(int(time.time() - start_time))))))
    print('#######################################')
