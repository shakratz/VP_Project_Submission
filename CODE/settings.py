def init():
    global input_name, stabilized_name, output_name, background_name, binary_video
    global alpha_name, trimap_name, extracted_name, matted_name, transformations
    global inputdir, outdir, tempdir, Stabilization_iterations, unstabilized_alpha_name
    global startFromFrameNum, maxCorners, qualityLevel, minDistance, blockSize

    inputdir = '../Input/' # The path to inputs
    input_name = inputdir + 'INPUT.avi'
    background_name = inputdir + 'background.jpg'

    outdir = '../Outputs/'  # The path to outputs
    output_name = outdir + 'OUTPUT.avi'
    stabilized_name = outdir + 'stabilize.avi'
    binary_video = outdir + 'binary.avi'
    alpha_name = outdir + 'alpha.avi'
    unstabilized_alpha_name = outdir + 'unstabilized_alpha.avi'
    trimap_name = outdir + 'trimap.avi'
    extracted_name = outdir + 'extracted.avi'
    matted_name = outdir + 'matted.avi'

    tempdir = '../Temp/' # The path to temp files
    transformations = tempdir + 'All_transformations.txt'

    # Parameters
    Stabilization_iterations = 4
    startFromFrameNum = 10  # tracking starting frame

    # Stabilization parameters
    maxCorners = 200
    qualityLevel = 0.01
    minDistance = 50
    blockSize = 5
