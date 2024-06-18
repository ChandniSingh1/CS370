# CS370
Treasure Hunt Game
Project Overview
Welcome to the Treasure Hunt Game project! In this game, the player competes against a pirate to find hidden treasure in a maze. The goal is to develop the intelligent agent (the pirate) using deep Q-learning to find the optimal path to the treasure.

This project includes two primary Python classes and a Jupyter notebook to guide you through the implementation:

TreasureMaze.py: Represents the game environment, with the maze defined as a matrix.
GameExperience.py: Stores game episodes, which include all states from the initial state to the terminal state, aiding the agent in learning through exploration.
You will complete the deep Q-learning implementation using a provided skeleton.

Getting Started
Prerequisites
Make sure you have the following Python packages installed:

numpy
tensorflow or keras
datetime
You can install these packages using pip:

bash
Copy code
pip install numpy tensorflow
Files Provided
TreasureMaze.py: Defines the game environment.
GameExperience.py: Manages experience replay for the agent.
TreasureHunt.ipynb: The Jupyter notebook containing the skeleton code for the deep Q-learning implementation.
Usage
TreasureMaze.py: This class includes methods to create and manage the maze environment, reset the maze, observe the current state, perform actions, and check game status.

GameExperience.py: This class handles storing and retrieving game episodes for training the agent. It includes methods for remembering episodes and getting training data.

TreasureHunt.ipynb: Contains the skeleton code for the deep Q-learning implementation. You will need to complete the qtrain function to train the pirate agent.

Implementation Details
qtrain Function
The qtrain function is the core of the deep Q-learning implementation. Here’s a step-by-step explanation:

Initialization:

Set exploration factor (epsilon), number of epochs (n_epoch), maximum memory (max_memory), and data size (data_size).
Initialize the game environment and experience replay object.
Training Loop:

For each epoch:
Randomly select a starting position for the agent and reset the maze.
Observe the initial state.
While the game is not over:
Choose an action either by exploration (random choice) or exploitation (using the model's predictions).
Perform the action and observe the new state, reward, and game status.
Store the episode in the experience replay object.
Train the neural network model using the stored episodes.
Update the win history and check the win rate.
Print the epoch details, including loss, number of episodes, win count, win rate, and elapsed time.
If the win rate exceeds a threshold and the model passes the completion check, training stops.
Completion Check:

Ensure the training has exhausted all free cells and the agent has won in all cases.
Helper Functions
format_time: Utility for printing readable time strings.
Example
Here’s a simple example to illustrate how to use the qtrain function:

python
Copy code
import numpy as np
from TreasureMaze import TreasureMaze
from GameExperience import GameExperience
from keras.models import Sequential
from keras.layers import Dense

# Define the maze as a numpy array
maze = np.array([
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1]
])

# Create a simple neural network model
model = Sequential()
model.add(Dense(24, input_shape=(4,), activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(4, activation='linear'))
model.compile(optimizer='adam', loss='mse')

# Train the agent using the qtrain function
qtrain(model, maze, n_epoch=1000, max_memory=500, data_size=50)

What Do Computer Scientists Do?

Computer scientists play a crucial role in shaping the modern world through their work in software development, algorithm design, data management, artificial intelligence, and cybersecurity. Their innovations drive technological advancements that improve quality of life, such as in healthcare with medical imaging and in everyday life with smart devices. They provide solutions to complex problems, enhance economic growth through job creation, and ensure information and services are more accessible globally. By securing systems against cyber threats, they protect data and maintain privacy, which is essential in the digital age.

Ethically, computer scientists have a responsibility to prioritize user privacy and security, ensuring robust protection against data breaches and cyber-attacks. They must be transparent about data usage, design inclusive and accessible systems, and develop reliable solutions that prioritize user safety. Considering the broader social impact of their work, they strive to create positive social outcomes and mitigate negative effects, such as job displacement due to automation. Maintaining professional integrity, avoiding conflicts of interest, and committing to continuous learning are essential to upholding ethical standards and advancing the field responsibly.
