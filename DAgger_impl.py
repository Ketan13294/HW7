import pybullet as p  # physics simulation API
import pybullet_data
import numpy as np
import os
import time
import pickle
from robot import Panda
from teleop import KeyboardController
import torch
from models import MLPPolicy
from torch.utils.data import Dataset, DataLoader

# MyData dataset wrapper for demonstration tuples saved as pickled list.
# Each entry is expected to be a concatenation of state (6 dims) and action (3 dims).
class MyData(Dataset):

    def __init__(self, loadname):
        # load previously recorded demonstrations from file
        self.data = pickle.load(open(loadname, "rb"))
        self.data = torch.FloatTensor(self.data)
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    
    def append(self, item):
        # append a single demo tuple (as list or array) to the dataset tensor
        self.data = torch.cat((self.data, torch.FloatTensor([item])), dim=0)


# train_model: behavior cloning training loop for the MLPPolicy
def train_model(train_data):

    # training parameters
    print("[-] training bc")
    EPOCH = 1000
    LR = 0.005

    # initialize model and optimizer
    model = MLPPolicy(state_dim=6, hidden_dim=64, action_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # prepare DataLoader from dataset
    BATCH_SIZE = int(len(train_data) / 10.)
    print("my batch size is:", BATCH_SIZE)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # main training loop: forward, compute MSE loss, backward, step
    for epoch in range(EPOCH+1):
        for batch, x in enumerate(train_set):
        
            # separate states and actions from the batch
            states = x[:, 0:6]
            actions = x[:, 6:9]
            actions_hat = model(states)

            # compute the loss between actual and predicted
            loss = model.mse_loss(actions, actions_hat)
                 
            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 100 == 0:
            # periodic logging and checkpointing
            print(epoch, loss.item())
            torch.save(model.state_dict(), "data/model_weights_updated")
        
    return model


# parameters
control_dt = 1. / 240.  # simulation control timestep
n_demos = 20             # number of demonstration episodes to run

# create simulation and place camera
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                cameraYaw=40.0,
                                cameraPitch=-30.0, 
                                cameraTargetPosition=[0.5, 0.0, 0.2])

# load the environment objects (plane, table, cube)
urdfRootPath = pybullet_data.getDataPath()
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
cube = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.5, 0, 0.025])

# load the robot with initial joint configuration
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
panda = Panda(basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                jointStartPositions=jointStartPositions)

teleop = KeyboardController()  # keyboard controller for toggling assistance/relabeling

# load an existing dataset and pretrained BC model
# dataset contains demonstrations; model predicts 3D delta position for end-effector
dataset = MyData("data/dataset_2_narrow.pkl")
model = MLPPolicy(state_dim=6, hidden_dim=64, action_dim=3)
model.load_state_dict(torch.load('data/model_weights_2_narrow'))
model.eval()

action_magnitude = 0.1  # clip magnitude for safety
button_timer = 0        # debouncing timer for keyboard input

# run multiple demonstration episodes; user can toggle "retrain" mode to relabel trajectories
for demo_idx in range(n_demos):
    print("Demo episode:", demo_idx)
    data = []               # collect new state-action tuples when in relabel mode
    retrain_toggle = False  # when True, use expert (cube->robot) action instead of policy

    panda.reset(jointStartPositions)
    # randomize target cube position within a bounding box on the table
    cube_position = np.random.uniform([0.2, -0.3, 0.025], [0.6, +0.3, 0.025])
    p.resetBasePositionAndOrientation(cube, cube_position, p.getQuaternionFromEuler([0, 0, 0]))


    # run sequence of control steps for this episode
    for time_idx in range (1000):
        # get the robot's end-effector position
        robot_state = panda.get_state()
        robot_pos = np.array(robot_state["ee-position"])

        # if relabeling (expert), compute action as vector from robot to cube; otherwise use policy
        if retrain_toggle:
            action = cube_position - robot_pos
        else:
            state = torch.FloatTensor(robot_pos.tolist() + cube_position.tolist())
            action = model(torch.FloatTensor(state)).detach().numpy()
        
        # clip large actions to a max magnitude
        if np.linalg.norm(action) > action_magnitude:
            action *= action_magnitude / np.linalg.norm(action)

        # store the state-action pair when collecting expert data (relabel mode)
        if retrain_toggle:
            state = robot_pos.tolist() + cube_position.tolist()
            data.append(state + action.tolist())            

        input = teleop.get_action()  # read keyboard input

        if input[7] == +1 and time.time()-button_timer > 0.5: # "R" key toggles assistance/relabel
            retrain_toggle = not retrain_toggle
            button_timer = time.time()

        # execute the chosen action on the robot and step the simulator
        panda.move_to_pose(robot_pos + action, ee_rotz=0, positionGain=0.01)
        p.stepSimulation()
        time.sleep(control_dt)

    # if expert data was collected, append to dataset and retrain the BC model (DAgger-style)
    if retrain_toggle:
        print("[-] retraining model with new data")
        for item in data:
            dataset.append(item)
        model = train_model(dataset)
        model.eval()
        retrain_toggle = False
