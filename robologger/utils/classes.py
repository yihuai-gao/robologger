from enum import Enum

class Morphology(Enum):
    """
    Morphology of the robot.
    - Single Arm
    - Bi-Manual (optional head / torso)
    - Wheel-Based Single Arm
    - Wheel-Based Bi-Manual (optional head / torso)
    - Humanoid (optional head / torso)
    # NOTE: quadruped not yet supported
    """
    SINGLE_ARM = "single_arm"
    BI_MANUAL = "bi_manual"
    WHEEL_BASED_SINGLE_ARM = "wheel_based_single_arm"
    WHEEL_BASED_BI_MANUAL = "wheel_based_bi_manual"
    HUMANOID = "humanoid"

class CameraName(Enum):
    """
    Possible name choices for cameras.
    - right_wrist_camera_0, 
    - left_wrist_camera_0, ...
    - head_camera_0, ...
    - body_camera_0, ...
    - third_person_camera_0, ...
    NOTE: if there are multiple cameras of the same type, the index will be appended.
    """
    RIGHT_WRIST_CAMERA_ = "right_wrist_camera_"
    LEFT_WRIST_CAMERA_ = "left_wrist_camera_"
    HEAD_CAMERA_ = "head_camera_"
    BODY_CAMERA_ = "body_camera_"
    THIRD_PERSON_CAMERA_ = "third_person_camera_"

class RobotName(Enum):
    """
    Possible robot/gripper names.
    - right_arm
    - left_arm
    - head
    - body
    - left_end_effector
    - right_end_effector
    """
    RIGHT_ARM = "right_arm"
    LEFT_ARM = "left_arm"
    HEAD = "head"
    BODY = "body"
    LEFT_END_EFFECTOR = "left_end_effector"
    RIGHT_END_EFFECTOR = "right_end_effector"

class EndEffectorSetup(Enum):
    """
    End effector setups.
    - ARX+Finray
    - WSG50+Finray
    - Robotiq 
    """
    ARX_FINRAY = "arx_finray"
    WSG50_FINRAY = "wsg50_finray"
    ROBOTIQ = "robotiq"

if __name__ == "__main__":
    # test the enum names
    for name in Morphology:
        print(name.value)
    for name in CameraName:
        print(name.value)
    for name in RobotName:
        print(name.value)
    for name in EndEffectorSetup:
        print(name.value)