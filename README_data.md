# Data Explanation

## Overview

This data packet contains three parts: 

2. Face related information can be found in `Face_Features/***.csv` files. You can find more detailed information in later this file. Presentation and rest part are separated. You can also find the time stamp in the file name. The file name rule is: `Name-Class(Present or Rest)_YYYY-MM-DD_HH_MM`
3. Eda are exported to csv files, it contains all of the bio-signals collected. You can use csv file to do data analysis in python/Matlab. The separation and nomination of eda exported csv files are similar to face features csv files. 
3. Eda and face features are separated by presenting/resting status.
4. Eda and face feature video are using the different sample frequency(16 and 30), also the start and ending time may slightly different. But as you treat all these features as a whole series, you can simply ignore the trivial difference. Also, if you want to use a single model for all these data, maybe you need to consider resample.
5. You can find more information about the face feature extraction tool OpenFace [here](https://github.com/TadasBaltrusaitis/OpenFace).

## CSV Feature Files

Part of OpenFace processing outputs a comma separated value file of the following format:
```
timestamp, gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, p_scale, p_rx, p_ry, p_rz, p_tx, p_ty, AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU25_r, AU26_r, AU04_c, AU12_c, AU15_c, AU23_c, AU28_c, AU45_c
```

The header specifies the meaning of each column. The explanation of each:

**Basic**

`timestamp` the timer of video being processed in seconds (in case of sequences)

**Gaze related**

`gaze_0_x, gaze_0_y, gaze_0_z` Eye gaze direction vector in world coordinates for eye 0 (normalized), eye 0 is the leftmost eye in the image (think of it as a ray going from the left eye in the image in the direction of the eye gaze)

`gaze_1_x, gaze_1_y, gaze_1_z` Eye gaze direction vector in world coordinates for eye 1 (normalized), eye 1 is the rightmost eye in the image (think of it as a ray going from the right eye in the image in the direction of the eye gaze)

`gaze_angle_x, gaze_angle_y` Eye gaze direction in radians in world coordinates averaged for both eyes and converted into more easy to use format than gaze vectors. If a person is looking left-right this will results in the change of gaze_angle_x (from positive to negative) and, if a person is looking up-down this will result in change of gaze_angle_y (from negative to positive), if a person is looking straight ahead both of the angles will be close to 0 (within measurement error).

**Pose**

`pose_Tx, pose_Ty, pose_Tz` the location of the head with respect to camera in millimeters (positive Z is away from the camera)

`pose_Rx, pose_Ry, pose_Rz` Rotation is in radians around X,Y,Z axes with the convention `R = Rx >md.png COPYING Config.plist CopyAsMarkdown-demo.mp4 README.md _Signature.plist html2md.sh html2text.py Ry >md.png COPYING Config.plist CopyAsMarkdown-demo.mp4 README.md _Signature.plist html2md.sh html2text.py Rz`, left-handed positive sign. This can be seen as pitch (Rx), yaw (Ry), and roll (Rz). The rotation is in world coordinates with camera being the origin.

**Facial Action Units**

Facial Action Units (AUs) are a way to describe human facial expression, more details on Action Units can be found [here](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units)

The system can detect the intensity (from 0 to 5) of 17 AUs:

`AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r`

And the presense (0 absent, 1 present) of 18 AUs:

`AU01_c, AU02_c, AU04_c, AU05_c, AU06_c, AU07_c, AU09_c, AU10_c, AU12_c, AU14_c, AU15_c, AU17_c, AU20_c, AU23_c, AU25_c, AU26_c, AU28_c, AU45_c`



By Zhongyang Zhang