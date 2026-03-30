from fanuc_rmi import RobotClient
from take_images import take_images
import json

robot = RobotClient(host="192.168.1.22")
robot.connect()
robot.initialize(uframe=1, utool=1)
robot.speed_override(60)

# robot.read_joint_coordinates()
sequence_id = 1
with open("robot_position_joint.jsonl") as f:
    for line in f:
        data = json.loads(line)
        joint_angles = data["JointAngle"]
       
        robot.joint_absolute({"J1": joint_angles["J1"], "J2": joint_angles["J2"], "J3": joint_angles["J3"], "J4": joint_angles["J4"], "J5": joint_angles["J5"], "J6": joint_angles["J6"]},speed_percentage=20,sequence_id=sequence_id)
        robot.wait_time(2, sequence_id=sequence_id+1)    
        take_images(captures_dir="captures_camera",warmup_sec=1.5,read_timeout_ms=1000,discard_initial_frames=5,print_timings=True)
        sequence_id += 2
robot.close()
    
