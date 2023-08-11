import cv2
import mediapipe as mp


class PoseDetector():
    def __init__(
        self, static_image_mode: bool = False, model_complexity: int = 1, smooth_landmarks: bool = True,
        enable_segmentation: bool = False, smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            self.static_image_mode, self.model_complexity, self.smooth_landmarks,
            self.enable_segmentation, self.smooth_segmentation,
            self.min_detection_confidence, self.min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, image, draw: bool = True):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:  # type: ignore
            if draw:
                self.mp_draw.draw_landmarks(
                    image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,  # type: ignore
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(255, 255, 0), thickness=2, circle_radius=2
                    )

                )
        return image

    def get_postition(self, image, draw: bool = True):

        list_for_landmarks = []

        if self.results.pose_landmarks:  # type: ignore
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):  # type: ignore
                height, width, channel = image.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                list_for_landmarks.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

        return list_for_landmarks

    def track_certain_part(self, img, body_part: str = 'right_knee'):
        # TODO correst the order
        body_part_mapping = {
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'right_eye_inner': 3,
            'right_eye': 4,
            'left_ear': 5,
            'right_ear': 6,
            'mouth_left': 7,
            'mouth_right': 8,
            'left_shoulder': 9,
            'right_shoulder': 10,
            'left_elbow': 11,
            'right_elbow': 12,
            'left_wrist': 13,
            'right_wrist': 14,
            'left_pinky': 15,
            'right_pinky': 16,
            'left_index': 17,
            'right_index': 18,
            'left_thumb': 19,
            'right_thumb': 20,
            'left_hip': 21,
            'right_hip': 22,
            'left_knee': 23,
            'right_knee': 24,
            'left_ankle': 25,
            'right_ankle': 26,
            'left_heel': 27,
            'right_heel': 28,
            'left_foot_index': 29,
            'right_foot_index': 30
        }

        landmark_list = self.get_postition(img, draw=False)
        if len(landmark_list) != 0:
            cv2.circle(
                img,
                (
                    landmark_list[body_part_mapping[body_part]][1], landmark_list[body_part_mapping[body_part]][2]
                ), 10, (255, 0, 0), cv2.FILLED
            )
