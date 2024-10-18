import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

# Function to draw landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# landmark = [
#     PoseLandmark.NOSE,
#     PoseLandmark.LEFT_EYE,
#     PoseLandmark.LEFT_EYE_INNER,
#     PoseLandmark.LEFT_EYE_OUTER,
#     PoseLandmark.LEFT_EAR,
#     PoseLandmark.RIGHT_EAR,
#     PoseLandmark.LEFT_SHOULDER,
#     PoseLandmark.LEFT_ELBOW,
#     PoseLandmark.LEFT_WRIST,
#     PoseLandmark.LEFT_PINKY,
#     PoseLandmark.RIGHT_PINKY,
#     PoseLandmark.LEFT_INDEX,
#     PoseLandmark.RIGHT_INDEX,
#     PoseLandmark.LEFT_THUMB,
#     PoseLandmark.RIGHT_THUMB,
#     PoseLandmark.LEFT_HIP,
#     PoseLandmark.LEFT_KNEE,
#     PoseLandmark.LEFT_ANKLE,
#     PoseLandmark.LEFT_HEEL,
#     PoseLandmark.LEFT_FOOT_INDEX,
#     PoseLandmark.MOUTH_LEFT,
#     PoseLandmark.MOUTH_RIGHT
# ]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/pose_landmarker_heavy.task'),
    running_mode=VisionRunningMode.IMAGE)

with PoseLandmarker.create_from_options(options) as landmarker:

    # Load the input image.
    image = mp.Image.create_from_file("tests/image/test1.jpg")

    # Detect pose landmarks from the input image.
    detection_result = landmarker.detect(image)

    # Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # Display the annotated image.
    cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # # Create a Pose Landmarker object.
# # base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_heavy.task')
# # options = vision.PoseLandmarkerOptions(
# #     base_options=base_options,
# #     output_segmentation_masks=True
# # )
# # detector = vision.PoseLandmarker.create_from_options(options)

# # # Load the input image.
# # image = mp.Image.create_from_file("tests/image/test2.jpg")

# # # Detect pose landmarks from the input image.
# # detection_result = detector.detect(image)

# # # Process the detection result. In this case, visualize it.
# # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# # # Display the annotated image.
# # cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
