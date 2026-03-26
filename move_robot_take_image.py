from fanuc_rmi import RobotClient
from take_images import take_images
import json

robot = RobotClient(host="192.168.1.22")
robot.connect()
robot.initialize(uframe=1, utool=1)
robot.speed_override(20)


sequence_id = 1
with open("robot_position_joint.jsonl") as f:
    for line in f:
        data = json.loads(line)
        joint_angles = data["JointAngle"]
       
        robot.joint_absolute({"J1": joint_angles["J1"], "J2": joint_angles["J2"], "J3": joint_angles["J3"], "J4": joint_angles["J4"], "J5": joint_angles["J5"], "J6": joint_angles["J6"]},speed_percentage=20,sequence_id=sequence_id)
        take_images(captures_dir="captures_camera", warmup_sec=0.5, read_timeout_ms=200, print_timings=True)
        sequence_id += 1
robot.close()
