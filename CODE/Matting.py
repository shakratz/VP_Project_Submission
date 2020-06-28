import time
from datetime import timedelta
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg
from tqdm import tqdm
import auxiliary_functions as AF
import settings

def fromBinaryToTrimap(binary_video, output_video):
    # Getting Video parameters
    cap = cv2.VideoCapture(binary_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setting output file
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for fr in tqdm(range(frameCount)):
        # Reading a frame
        ret, frame = cap.read()
        # Adding a gray zone
        dilition_for_trimap = cv2.dilate(frame, (np.ones((20, 20), np.uint8)), iterations=1)
        erosion_for_trimap = cv2.erode(frame, (np.ones((25, 25), np.uint8)), iterations=2)
        frame[np.logical_and(dilition_for_trimap == 255, erosion_for_trimap == 0)] = 100

        frame = frame.astype(np.uint8)
        out.write(frame)

    # Releasing the videos
    out.release()
    cap.release()

def CalculateMatrices(img, mask=None, eps=10 ** (-7), win_rad=1):
    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    block = (win_diam, win_diam)
    shape = (indsM.shape[0] - block[0] + 1, indsM.shape[1] - block[1] + 1) + block
    strides = (indsM.strides[0], indsM.strides[1]) + indsM.strides
    win_inds = as_strided(indsM, shape=shape, strides=strides)

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    if mask is not None:
        mask = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((win_diam, win_diam), np.uint8)
        ).astype(np.bool)
        win_mask = np.sum(mask.ravel()[win_inds], axis=2)
        win_inds = win_inds[win_mask > 0, :]
    else:
        win_inds = win_inds.reshape(-1, win_size)

    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps / win_size) * np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1.0 / win_size) * (1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h * w, h * w))
    return L


def OneFrameTrimapToAlpha(image, trimap, trimap_c=100.0):
    consts_map = (trimap < 0.1) | (trimap > 0.9)
    trimap_c_map = trimap_c * consts_map

    Matrices_solution = CalculateMatrices(image, ~consts_map if consts_map is not None else None)

    confidence = scipy.sparse.diags(trimap_c_map.ravel())

    solution = scipy.sparse.linalg.spsolve(Matrices_solution + confidence, trimap.ravel() * trimap_c_map.ravel())
    alpha = np.minimum(np.maximum(solution.reshape(trimap.shape), 0), 1)
    return alpha


def fromTrimapToAlpha(stabilized_video, trimap_video, alpha_video):
    # Read videos
    cap_trimap = cv2.VideoCapture(trimap_video)
    cap_stab = cv2.VideoCapture(stabilized_video)

    # Get video parameters
    w = int(cap_trimap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_trimap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap_trimap.get(cv2.CAP_PROP_FOURCC))
    frameCount = int(cap_trimap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_trimap.get(cv2.CAP_PROP_FPS)

    # output video
    output_name = alpha_video
    out_alpha = cv2.VideoWriter(output_name, fourcc, fps, (w, h))

    for i in tqdm(range(frameCount)):
        # Read next frame
        success, trimap_frame = cap_trimap.read()
        success, stab_frame = cap_stab.read()

        # changing to gray
        trimap_frame = cv2.cvtColor(trimap_frame, cv2.COLOR_BGR2GRAY) / 255.0
        stab_frame = stab_frame / 255.0

        # resizing for better run time
        trimap_frame = cv2.resize(trimap_frame, (int(w / 2), int(h / 2)))
        stab_frame = cv2.resize(stab_frame, (int(w / 2), int(h / 2)))

        if not success:
            break
        # converts each frame
        output_frame = OneFrameTrimapToAlpha(stab_frame, trimap_frame)

        output_frame = output_frame * 255
        # resizing back to original dims
        output_frame = cv2.resize(output_frame, (int(w), int(h)))

        # changing to 3 channels (instructions)
        output_frame = cv2.merge([output_frame,output_frame,output_frame])

        output_frame = output_frame.astype(np.uint8)
        out_alpha.write(output_frame)

    cap_trimap.release()
    cap_stab.release()
    out_alpha.release()


def MattingOnNewBackground(Background_img, output_video, alpha_video, stabilized_video):
    # Read input
    cap_alpha = cv2.VideoCapture(alpha_video)
    cap_stab = cv2.VideoCapture(stabilized_video)
    background_image = cv2.imread(Background_img)

    # Get video parameters
    w = int(cap_alpha.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_alpha.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap_alpha.get(cv2.CAP_PROP_FOURCC))
    frameCount = int(cap_stab.get(cv2.CAP_PROP_FRAME_COUNT))
    frameCount = int(cap_alpha.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_alpha.get(cv2.CAP_PROP_FPS)

    # Resizing background image to fit video
    background_image = cv2.resize(background_image, (w, h), interpolation=cv2.INTER_AREA)

    # output video
    output_name = output_video
    out = cv2.VideoWriter(output_name, fourcc, fps, (w, h))
    output_frame = np.zeros((h, w, 3), np.float32)
    for i in tqdm(range(frameCount)):
        # Read next frame
        success, alpha_frame = cap_alpha.read()
        success, stab_frame = cap_stab.read()
        if not success:
            break
        alpha_gray = cv2.cvtColor(alpha_frame, cv2.COLOR_RGB2GRAY) / 255.0

        # Setting pixels value to background and foreground by alpha map
        output_frame[:, :, 0] = (1 - alpha_gray) * background_image[:, :, 0] + alpha_gray * stab_frame[:, :, 0]
        output_frame[:, :, 1] = (1 - alpha_gray) * background_image[:, :, 1] + alpha_gray * stab_frame[:, :, 1]
        output_frame[:, :, 2] = (1 - alpha_gray) * background_image[:, :, 2] + alpha_gray * stab_frame[:, :, 2]
        output_frame = output_frame.astype(np.uint8)
        out.write(output_frame)

    cap_alpha.release()
    cap_stab.release()


def Matting_Main(All_transformations):
    start_time = time.time()
    binary_video = settings.binary_video
    stabilized_video = settings.stabilized_name
    background_name = settings.background_name
    alpha_name = settings.alpha_name
    unstabilized_alpha_name = settings.unstabilized_alpha_name
    matted_name = settings.matted_name
    trimap_video = settings.trimap_name


    # converting binary to trimap
    print('# Sub-Matting: converting Binary to Trimap ###')
    fromBinaryToTrimap(binary_video, trimap_video)

    # converting trimap to alpha map
    print('# Sub-Matting: converting Trimap to Alpha map ###')
    fromTrimapToAlpha(stabilized_video, trimap_video,alpha_name)

    # Creating Unstabilized Alpha
    print('# Sub-Matting: Creating Unstabilized Alpha ###')
    AF.CreatingUnstabilizedAlpha(All_transformations, unstabilized_alpha_name)

    # matting object on new background
    print('# Sub-Matting: Matting on new background###')
    MattingOnNewBackground(background_name, matted_name, alpha_name, stabilized_video)
    print('### FINISHED Matting. Run time: {:0>8} ###'.format(str(timedelta(seconds=(int(time.time() - start_time))))))
    print('#######################################')
