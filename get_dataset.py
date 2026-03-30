import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import pickle
from robot import Panda


# parameters
control_dt = 1. / 240.

# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                cameraYaw=40.0,
                                cameraPitch=-30.0, 
                                cameraTargetPosition=[0.5, 0.0, 0.2])

# load the objects
urdfRootPath = pybullet_data.getDataPath()
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
cube = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.5, 0, 0.025])

# load the robot
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)


# collect the demonstrations
# these demonstrations move from the robot's home position to the cube position
for dataset_type in  ["narrow","wide"]: # "narrow" or "wide"
    for n_demos in [1,10,20]:
        dataset = []
        action_magnitude = 0.1
        for demo_idx in range(n_demos):

            # reset the robot
            panda.reset(jointStartPositions)
            if dataset_type == "narrow":
                cube_position = np.random.uniform([0.3, -0.1, 0.025], [0.5, +0.1, 0.025])
            elif dataset_type == "wide":
                cube_position = np.random.uniform([0.2, -0.3, 0.025], [0.6, +0.3, 0.025])
            p.resetBasePositionAndOrientation(cube, cube_position, p.getQuaternionFromEuler([0, 0, 0]))

            # run sequence of position and gripper commands
            for time_idx in range (1000):

                # get the robot's position
                robot_state = panda.get_state()
                robot_pos = np.array(robot_state["ee-position"])

                # select the robot's action
                action = cube_position - robot_pos
                if np.linalg.norm(action) > action_magnitude:
                    action *= action_magnitude / np.linalg.norm(action)

                # store the state-action pair
                state = robot_pos.tolist() + cube_position.tolist()
                dataset.append(state + action.tolist())

                # move the robot with action
                panda.move_to_pose(robot_pos + action, ee_rotz=0, positionGain=0.01)
                p.stepSimulation()
                time.sleep(control_dt)

        # save the dataset of demonstrations
        filename = "data/dataset_"+ str(n_demos) + "_" + str(dataset_type) +".pkl"
        pickle.dump(dataset, open(filename, "wb"))
        print("dataset has this many state-action pairs:", len(dataset))