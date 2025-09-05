from enum import StrEnum, auto

class Morphology(StrEnum):
    """
    Morphology of the robot.
    - Single Arm
    - Bi-Manual (optional head / torso)
    - Wheel-Based Single Arm
    - Wheel-Based Bi-Manual (optional head / torso)
    - Humanoid (optional head / torso)
    # NOTE: quadruped not yet supported
    """
    SINGLE_ARM = auto()
    BI_MANUAL = auto()
    WHEEL_BASED_SINGLE_ARM = auto()
    WHEEL_BASED_BI_MANUAL = auto()
    HUMANOID = auto()

class CameraName(StrEnum):
    """
    Possible name choices for cameras.
    - right_wrist_camera_0, 
    - left_wrist_camera_0, ...
    - head_camera_0, ...
    - body_camera_0, ...
    - third_person_camera_0, ...
    NOTE: if there are multiple cameras of the same type, the index will be appended.
    """
    RIGHT_WRIST_CAMERA_ = auto()
    LEFT_WRIST_CAMERA_ = auto()
    HEAD_CAMERA_ = auto()
    BODY_CAMERA_ = auto()
    THIRD_PERSON_CAMERA_ = auto()

class EndEffectorSetup(StrEnum):
    """
    End effector setups.
    - ARX+Finray
    - WSG50+Finray
    - Robotiq 
    """
    ARX_FINRAY = auto()
    WSG50_FINRAY = auto()
    ROBOTIQ = auto()

if __name__ == "__main__":
    # test the enum names
    for name in Morphology:
        print(name)
    for name in CameraName:
        print(name)
    for name in EndEffectorSetup:
        print(name)