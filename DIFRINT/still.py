
import os
import cv2

frame = cv2.imread('/data4/pengzhan/work/DIFRINT_plus/still_0.png')
for count in range(100):
    filename = os.path.sep.join(['/data4/pengzhan/work/DIFRINT_plus/short/0', '0'+'_frame'+"%05d.png"%(count)])
    cv2.imwrite(filename, frame)