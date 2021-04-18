import datetime

# 日志处理
def get_log_data(type, angle_data, index):
    # 时间
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 腋下角度
    left_armpit_angle = angle_data['armpit']['left_armpit_angle']
    if left_armpit_angle == -1:
        left_armpit_angle = "未识别"
    right_armpit_angle = angle_data['armpit']['right_armpit_angle']
    if right_armpit_angle == -1:
        right_armpit_angle = "未识别"
    armpit = str(left_armpit_angle) + '/' + str(right_armpit_angle)

    # 手肘角度
    left_elbow_angle = angle_data['elbow']['left_elbow_angle']
    if left_elbow_angle == -1:
        left_elbow_angle = "未识别"
    right_elbow_angle = angle_data['elbow']['right_elbow_angle']
    if right_elbow_angle == -1:
        right_elbow_angle = "未识别"
    elbow = str(left_elbow_angle) + '/' + str(right_elbow_angle)

    # 腰部角度
    left_hip_angle = angle_data['hip']['left_hip_angle']
    if left_hip_angle == -1:
        left_hip_angle = "未识别"
    right_hip_angle = angle_data['hip']['right_hip_angle']
    if right_hip_angle == -1:
        right_hip_angle = "未识别"
    hip = str(left_hip_angle) + '/' + str(right_hip_angle)

    # 膝盖角度
    left_knee_angle = angle_data['knee']['left_knee_angle']
    if left_knee_angle == -1:
        left_knee_angle = "未识别"
    right_knee_angle = angle_data['knee']['right_knee_angle']
    if right_knee_angle == -1:
        right_knee_angle = "未识别"
    knee = str(left_knee_angle) + '/' + str(right_knee_angle)

    # 日志数组
    log = [now_time, type, index, armpit, elbow, hip, knee]
    return log