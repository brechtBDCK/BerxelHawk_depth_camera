"""
A camera on a robot arm took depth images, 4 images for each paprika (a grey and blue one)
The position of the robot arm from where the images were taken is known, in robot_position_joint.jsonl

so eg for the grey paprika, we have 4 depth images taken from 4 different positions of the robot arm, and we have the corresponding robot arm joint positions for each of those 4 images.
This script reads the 4 point clouds generated from the 4 depth images, and concatenates them into a single point cloud for the grey paprika, and does the same for the blue paprika.
The output is 2 point clouds, one for the grey paprika and one for the blue paprika

"""
