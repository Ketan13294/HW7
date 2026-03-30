# import pybullet_data
# import numpy as np
# import os
# import time
# import pickle
# from robot import Panda
# from teleop import KeyboardController
# import torch
# from models import MLPPolicy
# from torch.utils.data import Dataset, DataLoader

# # import dataset for training
# class MyData(Dataset):

#     def __init__(self, loadname):
#         self.data = pickle.load(open(loadname, "rb"))
#         self.data = torch.FloatTensor(self.data)
#         print("imported dataset of length:", len(self.data))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self,idx):
#         return self.data[idx]
    
#     def append(self, item):
#         self.data = torch.cat((self.data, torch.FloatTensor([item])), dim=0)


# # train model
# def train_model(train_data):

#     # training parameters
#     print("[-] training bc")
#     EPOCH = 400
#     LR = 0.001

#     # initialize model and optimizer
#     model = MLPPolicy(state_dim=6, hidden_dim=64, action_dim=3)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     # initialize dataset
#     BATCH_SIZE = int(len(train_data) / 10.)
#     print("my batch size is:", BATCH_SIZE)
#     train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

#     # main training loop
#     for epoch in range(EPOCH+1):
#         for batch, x in enumerate(train_set):
        
#             # collect the demonstrated states and actions
#             states = x[:, 0:6]
#             actions = x[:, 6:9]
#             actions_hat = model(states)

#             # compute the loss between actual and predicted
#             loss = model.mse_loss(actions, actions_hat)
                 
#             # update model parameters
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#         if epoch % 100 == 0:
#             print(epoch, loss.item())
#             torch.save(model.state_dict(), "data/model_weights_updated")
        
#     return model


# # parameters
# control_dt = 1. / 240.
# n_demos = 20

# # create simulation and place camera
# physicsClient = p.connect(p.GUI)
# p.setGravity(0, 0, -9.81)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.resetDebugVisualizerCamera(cameraDistance=1.0, 
#                                 cameraYaw=40.0,
#                                 cameraPitch=-30.0, 
#                                 cameraTargetPosition=[0.5, 0.0, 0.2])

# # load the objects
# urdfRootPath = pybullet_data.getDataPath()
# plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
# table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
# cube = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.5, 0, 0.025])

# # load the robot
# jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
# panda = Panda(basePosition=[0, 0, 0],
#                 baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
#                 jointStartPositions=jointStartPositions)

# teleop = KeyboardController()

# # collect the demonstrations
# # these demonstrations move from the robot's home position to the cube position
# dataset = MyData("data/dataset_10_narrow.pkl")
# model = MLPPolicy(state_dim=6, hidden_dim=64, action_dim=3)
# model.load_state_dict(torch.load('data/model_weights_10_wide'))
# model.eval()

# action_magnitude = 0.1
# button_timer = 0

# for demo_idx in range(n_demos):
#     data = []
#     retrain_toggle = False

#     panda.reset(jointStartPositions)
#     cube_position = np.random.uniform([0.2, -0.3, 0.025], [0.6, +0.3, 0.025])
#     p.resetBasePositionAndOrientation(cube, cube_position, p.getQuaternionFromEuler([0, 0, 0]))


#     # run sequence of position and gripper commands
#     for time_idx in range (1000):
#         # get the robot's position
#         robot_state = panda.get_state()
#         robot_pos = np.array(robot_state["ee-position"])

#         if retrain_toggle:
#             action = cube_position - robot_pos
#         else:
#             state = torch.FloatTensor(robot_pos.tolist() + cube_position.tolist())
#             action = model(torch.FloatTensor(state)).detach().numpy()
        
#         # select the robot's action
#         if np.linalg.norm(action) > action_magnitude:
#             action *= action_magnitude / np.linalg.norm(action)

#         # store the state-action pair
#         if retrain_toggle:
#             state = robot_pos.tolist() + cube_position.tolist()
#             data.append(state + action.tolist())            

#         input = teleop.get_action()

#         if input[7] == +1 and time.time()-button_timer > 0.5: # "R" key
#             # Each press toggles the assistance mode on and off
#             retrain_toggle = not retrain_toggle
#             button_timer = time.time()

#         # move the robot with action
#         panda.move_to_pose(robot_pos + action, ee_rotz=0, positionGain=0.01)
#         p.stepSimulation()
#         time.sleep(control_dt)

#     if retrain_toggle:
#         print("[-] retraining model with new data")
#         for item in data:
#             dataset.append(item)
#         model = train_model(dataset)
#         model.eval()
#         retrain_toggle = False