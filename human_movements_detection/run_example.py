import cv2
from pose_model import PoseDetector


cap = cv2.VideoCapture('videos/2.mp4')

detector = PoseDetector()

while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    landmark_list = detector.get_postition(img)
    detector.track_certain_part(img, 'right_knee')
    print(landmark_list)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
