from Stabilize_video import Stabilize_Video_Main
from Background_Subtraction import backgroundSubtraction_Main
from Matting import Matting_Main
from Tracking_openCV import Tracking_Main_with_openCV
from Tracking import Tracking_Main
import numpy as np
import settings

# Initialize global variables
settings.init()

# #### Manual Selections ####### #
# Skip stabilization video (1-stabilize, 0-skip):
Stabilize = 1
# Manual tracking rectangle selection (1- manual, 0- automatic)
ManualTracking = 0


# ################# Stabilization ####################
# Input: Unstabilized video
# Output: Stabilized video
if Stabilize:
    All_transformations = Stabilize_Video_Main()
else:
    new_data = np.loadtxt(settings.transformations)
    All_transformations = new_data.reshape((4,205,3))

# ################# Background Subtraction ####################
# Input: Stabilized
# Output: Binary, Extracted
backgroundSubtraction_Main()

# ################# Matting ####################
# Input: Binary, Background image
# Output: Alpha, unstabilized alpha, trimap, matted
Matting_Main(All_transformations)

# ################# Tracking ####################
# Input: Alpha, Stabilized, Background image
# Output: OUTPUT(tracked)
Tracking_Main(ManualTracking)
#Tracking_Main_with_openCV(ManualTracking)  # Uncomment for tracking option 2






