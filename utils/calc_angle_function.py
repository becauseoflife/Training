import math


def calc_angle(landmark_points, visibility_th=0.5):
    # pose landmarks ： https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png

    # body_angle = []
    body_angles_json = {}

    # 手臂和身体的夹角，公共点 肩部
    left_armpit_angle = -1
    right_armpit_angle = -1
    if landmark_points[14][0] > visibility_th and landmark_points[24][0] and landmark_points[12][0] > visibility_th:
        left_armpit_angle = angle(landmark_points[14][1], landmark_points[24][1], landmark_points[12][1])
    if landmark_points[13][0] > visibility_th and landmark_points[23][0] > visibility_th and landmark_points[11][
        0] > visibility_th:
        right_armpit_angle = angle(landmark_points[13][1], landmark_points[23][1], landmark_points[11][1])
    # body_angles.append(['armpit', (left_armpit_angle, right_armpit_angle)])
    body_angles_json['armpit'] = {'left_armpit_angle': left_armpit_angle, 'right_armpit_angle': right_armpit_angle}

    # 手臂和肩膀的夹角，公共点 肩部
    left_shoulder_angle = -1
    right_shoulder_angle = -1
    if landmark_points[11][0] > visibility_th and landmark_points[14][0] > visibility_th and landmark_points[12][
        0] > visibility_th:
        left_shoulder_angle = angle(landmark_points[11][1], landmark_points[14][1], landmark_points[12][1])
    if landmark_points[12][0] > visibility_th and landmark_points[13][0] > visibility_th and landmark_points[11][
        0] > visibility_th:
        right_shoulder_angle = angle(landmark_points[12][1], landmark_points[13][1], landmark_points[11][1])
    # body_angles.append(['shoulder', (left_shoulder_angle, right_shoulder_angle)])
    body_angles_json['shoulder'] = {'left_shoulder_angle': left_shoulder_angle,
                                    'right_shoulder_angle': right_shoulder_angle}

    # 肩膀和手腕的夹角，公共点 肘部
    left_elbow_angle = -1
    right_elbow_angle = -1
    if landmark_points[12][0] > visibility_th and landmark_points[16][0] > visibility_th and landmark_points[14][
        0] > visibility_th:
        left_elbow_angle = angle(landmark_points[12][1], landmark_points[16][1], landmark_points[14][1])

    if landmark_points[11][0] > visibility_th and landmark_points[15][0] > visibility_th and landmark_points[13][
        0] > visibility_th:
        right_elbow_angle = angle(landmark_points[11][1], landmark_points[15][1], landmark_points[13][1])
    # body_angles.append(['elbow', (left_elbow_angle, right_elbow_angle)])
    body_angles_json['elbow'] = {'left_elbow_angle': left_elbow_angle, 'right_elbow_angle': right_elbow_angle}

    # 身体和大腿的夹角，公共点 臀部
    left_hip_angle = -1
    right_hip_angle = -1
    if landmark_points[12][0] > visibility_th and landmark_points[26][0] > visibility_th and landmark_points[24][
        0] > visibility_th:
        left_hip_angle = angle(landmark_points[12][1], landmark_points[26][1], landmark_points[24][1])
    if landmark_points[11][0] > visibility_th and landmark_points[25][0] > visibility_th and landmark_points[23][
        0] > visibility_th:
        right_hip_angle = angle(landmark_points[11][1], landmark_points[25][1], landmark_points[23][1])
    # body_angles.append(['hip', (left_hip_angle, right_hip_angle)])
    body_angles_json['hip'] = {'left_hip_angle': left_hip_angle, 'right_hip_angle': right_hip_angle}

    # 大腿和小腿之间的夹角，公共点 膝盖
    left_knee_angle = -1
    right_knee_angle = -1
    if landmark_points[24][0] > visibility_th and landmark_points[28][0] > visibility_th and landmark_points[26][
        0] > visibility_th:
        left_knee_angle = angle(landmark_points[24][1], landmark_points[28][1], landmark_points[26][1])
    if landmark_points[23][0] > visibility_th and landmark_points[27][0] > visibility_th and landmark_points[25][
        0] > visibility_th:
        right_knee_angle = angle(landmark_points[23][1], landmark_points[27][1], landmark_points[25][1])
    # body_angles.append(['knee', (left_knee_angle, right_knee_angle)])
    body_angles_json['knee'] = {'left_knee_angle': left_knee_angle, 'right_knee_angle': right_knee_angle}

    # body_angle.append(body_angles_json)
    # print(body_angles_json)
    return body_angles_json


def angle(point1, point2, public_point):
    x1 = point1[0] - public_point[0]
    y1 = point1[1] - public_point[1]
    x2 = point2[0] - public_point[0]
    y2 = point2[1] - public_point[1]

    # atan2 返回给定的 X 及 Y 坐标值的反正切值。
    angle1 = math.atan2(y1, x1)
    angle1 = int(angle1 * 180 / math.pi)

    angle2 = math.atan2(y2, x2)
    angle2 = int(angle2 * 180 / math.pi)

    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle

    return included_angle


