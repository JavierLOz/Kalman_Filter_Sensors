import cv2
import numpy as np
from ultralytics import YOLO

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                    [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.5

    def predict(self, coords_x, coords_y):
        ''' This function estimates the position of the object'''
        # Average the measurements from the three sensors (you can use a weighted average too)
        avg_x = np.mean(coords_x)
        avg_y = np.mean(coords_y)

        # Create a measurement array
        measurement = np.array([[np.float32(avg_x)],
                                [np.float32(avg_y)]])
        predicted = self.kf.predict()
        self.kf.correct(measurement)
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

# Load Kalman filter 
kf = KalmanFilter()

video_file = "./videoCarrito.mp4"
yolo_model = r"./models/yolo11n.pt"

# Load YOLOv8 model
model = YOLO(yolo_model)
model.fuse()

# D`efine detection parameters
conf_thres = 0.025
iou_thres = 0
max_det = 500
classes = ([36,41])
device = "cpu"

# Load video 
video = cv2.VideoCapture(video_file)

low_bounds = (0,170,140)
high_bounds = (30,255,255)
last_pose = (0,0)

while True:
    # Load frame from video
    ret,frame = video.read()
    
    # Make prediction over frame
    results = model.track(  source=frame,
                            conf=conf_thres,
                            iou=iou_thres,
                            max_det=max_det,
                            classes=classes,
                            device=device,
                            verbose=False,
                            persist=True,
                            retina_masks=False)

    # Plot prediction  
    plotted_image = results[0].plot()

    # Extract predition values 
    bounding_box = results[0].boxes.xywh
    classes = results[0].boxes.cls
    conf_bbox = results[0].boxes.conf

    # Get the position of the prediction with the highest confidence
    max = 0.0
    for box,conf in zip(bounding_box,conf_bbox):
        if conf > max:
            yolo_x,yolo_y,yolo_w,yolo_h = box[0],box[1],box[2],box[3]
            yolo_pose = (yolo_x,yolo_y)
            max = conf

    cv2.circle(plotted_image, (int(yolo_x),int(yolo_y)), 5, (0, 255, 0), 3) # draw magenta circle

    # Transform bgr frame to hsv 
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Apply color mask 
    mask_frame = cv2.inRange(frame,low_bounds,high_bounds)

    # Calculate centroid as average of the white pixels in the mask
    segmentation = np.where(mask_frame == 255)
    cx = np.mean(segmentation[1]) 
    cy = np.mean(segmentation[0]) 
    
    color_pose = (np.mean([cx,last_pose[0]]),np.mean([cy,last_pose[1]]))

    # Draw average centroid
    cv2.circle(plotted_image, (int(color_pose[0]),int(color_pose[1])), 5, (255, 255, 0), 3) # draw magenta circle
    mask_frame = cv2.GaussianBlur(mask_frame, (5, 5), 1.5)
    h, w = mask_frame.shape[:]

    circles = cv2.HoughCircles(mask_frame, cv2.HOUGH_GRADIENT, 1.5, int(w / 20), param1=50, param2=20, minRadius=15,
    maxRadius=20)

    prediction_coords_x = [yolo_pose[0],color_pose[0]]
    prediction_coords_y = [yolo_pose[1],color_pose[1]]

    if circles is not None :
        sorted_circles = sorted(circles[0, :],key=lambda x:(x[0],x[1]))
        for c in circles[0, :]:
            cv2.circle(frame, (int(c[0]), int(c[1])), int(c[2]), (0, 255, 0), 3)
            cv2.circle(frame, (int(c[0]), int(c[1])), 1, (0, 0, 255), 5)
        if len(sorted_circles) > 1:
            # get the middle point between the circles
            x_middle = (sorted_circles[1][0] + sorted_circles[0][0])/2
            y_middle = (sorted_circles[1][1] + sorted_circles[0][1])/2
            circles_pose = (x_middle,y_middle)
        else:
            circles_pose = sorted_circles[0]
        cv2.circle(plotted_image, (int(circles_pose[0]), int(circles_pose[1])), 5, (255, 0, 0), 3)

    
        prediction_coords_x = [yolo_pose[0],color_pose[0],circles_pose[0]]
        prediction_coords_y = [yolo_pose[1],color_pose[1],circles_pose[1]]

    # Make prediction
    predicted = kf.predict(np.array(prediction_coords_x),
                           np.array(prediction_coords_y))

    # Draw average centroid
    cv2.circle(plotted_image, (int(predicted[0]),int(predicted[1])), 5, (0, 0, 255), 3) # draw magenta circle

    cv2.imshow("frame",plotted_image)
    cv2.waitKey(25) 

    # Store last pose to smooth out result
    last_pose = (cx,cy)


# while True: 
#     # Load frame from video
#     ret,frame = video.read()
#     #  Transform bgr frame to hsv 
#     hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

#     # Apply color mask 
#     mask_frame = cv2.inRange(frame,low_bounds,high_bounds)
#     mask_frame = cv2.GaussianBlur(mask_frame, (5, 5), 1.5)

#     h, w = mask_frame.shape[:]

#     circles = cv2.HoughCircles(mask_frame, cv2.HOUGH_GRADIENT, 1.5, int(w / 20), param1=50, param2=20, minRadius=15,
#     maxRadius=20)

#     if circles is not None :
#         sorted_circles = sorted(circles[0, :],key=lambda x:(x[0],x[1]))
#         for c in circles[0, :]:
#             cv2.circle(frame, (int(c[0]), int(c[1])), int(c[2]), (0, 255, 0), 3)
#             cv2.circle(frame, (int(c[0]), int(c[1])), 1, (0, 0, 255), 5)
#         if len(sorted_circles) > 1:
#             # get the middle point between the circles
#             x_middle = (sorted_circles[1][0] + sorted_circles[0][0])/2
#             y_middle = (sorted_circles[1][1] + sorted_circles[0][1])/2
#             circles_pose = (x_middle,y_middle)
#         else:
#             circles_pose = sorted_circles[0]
#         cv2.circle(frame, (int(circles_pose[0]), int(circles_pose[1])), 5, (255, 0, 0), 3)
    

#     cv2.imshow("frame",frame)
#     cv2.waitKey(25) 


# # backSub = cv2.createBackgroundSubtractorMOG2(100, 300, False) # background subtractor, 100 frame history, 300 sensitivity, no shadows
#  # arm and face color mask
# frame_arm = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (0, 0, 151), (180, 222, 255))

# # foreground/movement mask
# fg_mask = backSub.apply(frame)

# # movement minus arm should leave only phone
# frame_final = fg_mask - frame_arm if ARM_ENABLED else fg_mask

#  # centroid is calculated as the average of all white pixels in the final mask
# segmentation = np.where(frame_final == 255)
# cx = np.mean(segmentation[1]) 
# cy = np.mean(segmentation[0]) 

# #circels part

# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
#                             param1=100, param2=30,
#                             minRadius=1, maxRadius=20) if HOUGH_ENABLED else None

# circ_found = False

# if circles is not None: # if any circles were found
#     circ_found = True
#     n_circ = 0 # number of circles
#     avg_circ_x = 0
#     avg_circ_y = 0
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         n_circ += 1
#         avg_circ_x += i[0]
#         avg_circ_y += i[1]
#         center = (i[0], i[1])
#         radius = i[2]
#         cv2.circle(frame, center, radius, (255, 0, 255), 3) # draw magenta circle
#     avg_circ_x /= n_circ # avg circle center x position
#     avg_circ_y /= n_circ # avg circle center y position