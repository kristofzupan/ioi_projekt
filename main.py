import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'pose_landmarker_full.task'


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS, solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def calculate_distance(poses):
    pose_landmarks_list = poses.pose_landmarks
    # Camera calibration (replace with your calibration data)
    focal_length = (1, 1)  # focal length in pixels
    principal_point = (1, 1)  # principal point in pixels

    for i in range(0, len(pose_landmarks_list)):
        # Example 2D pose coordinates (replace with your actual pose data)
        # pose_2d = np.array([(x1, y1), (x2, y2), ...])
        coordinates = []
        for j in range(0, len(pose_landmarks_list[i])):
            point = np.array([pose_landmarks_list[i][j].x * focal_length[0] + principal_point[0], pose_landmarks_list[i][j].y * focal_length[1] + principal_point[1], pose_landmarks_list[i][j].z])
            coordinates.append(point)

        numpy_cords = np.asarray(coordinates)
        # Calculate distances for each keypoint
        distances = np.linalg.norm(numpy_cords, axis=1)

        # Calculate the average distance
        average_distance = np.mean(distances)

        print(i+1, "- Average Distance to camera:", average_distance)


def main():
    cv2.namedWindow("test")
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        start_time = time.time()
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True, num_poses=10)
        detector = vision.PoseLandmarker.create_from_options(options)

        image_numpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # STEP 3: Load the input image.
        #image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)
        image = mp.Image.create_from_file("test.jpg")

        # STEP 4: Detect pose landmarks from the input image.
        detection_result = detector.detect(image)

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

        calculate_distance(detection_result)
        print("Detection time:", time.time() - start_time)

        cv2.imshow("test", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(20)

        rval, frame = vc.read()


if __name__ == "__main__":
    main()
