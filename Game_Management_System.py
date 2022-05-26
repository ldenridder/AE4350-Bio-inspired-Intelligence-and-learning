'''
Created on August 28, 2021
@author: Luc den Ridder
'''
#%%
from Transition_Learning_Agent import Transition_Learning_Agent
from Lib import play_games, display_results, plot_results, plot_sensitivity, plot_comparison, plot_comparison_2, plot_comparison_3, plot_heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Board_Environment import Board_Environment

# Hyperparameters
LEARNING_RATES = [.0005, 0.001, 0.005]
DISCOUNT_RATES = [.4,.45,.5]
LEARNING_RATE = LEARNING_RATES[0]
DISCOUNT_RATE = DISCOUNT_RATES[1]
RANDOM_ACTION_PROBABILITY = [.3,.3,.4]
ACTION_LIMIT = 300

ROUNDS = 20
TRAINING_EPISODES = 40
VALIDATION_EPISODES = 10

LUMBER = 4
BRICK = 4
WOOL = 2
GRAIN = 2
ORE = 0

REWARD_VALUE_1 = .5
REWARD_VALUE_2 = 2
REWARD_VALUE_3 = 3
REWARD_VALUE_4 = 0.2
REWARD_VALUE_5 = 0.1

sensitivity_training_results = []
sensitivity_validation_results = []
sensitivity_testing_results = []

# in case of sensitivity analysis
for i in range(1):
    # Learning agents are generated
    PLAYER1 = Transition_Learning_Agent(LEARNING_RATE, DISCOUNT_RATE, WOOL, BRICK, LUMBER, GRAIN, ORE, RANDOM_ACTION_PROBABILITY[i], REWARD_VALUE_1, REWARD_VALUE_2, REWARD_VALUE_3, REWARD_VALUE_4, REWARD_VALUE_5)
    PLAYER2 = Transition_Learning_Agent(LEARNING_RATE, DISCOUNT_RATE, WOOL, BRICK, LUMBER, GRAIN, ORE, RANDOM_ACTION_PROBABILITY[i], REWARD_VALUE_1, REWARD_VALUE_2, REWARD_VALUE_3, REWARD_VALUE_4, REWARD_VALUE_5)
    PLAYER3 = Transition_Learning_Agent(0, 0, WOOL, BRICK, LUMBER, GRAIN, ORE, 1, 0, 0, 0, 0, 0)
    
    
    all_training_results = []
    all_validation_results = []
    all_training_reward_values = []
    all_validation_reward_values = []
    all_training_board_spread = []
    all_validation_board_spread = []
    visited_states = []
    
    for j in range(ROUNDS):
      # Games in training are played
      PLAYER1.set_random_action_probability(RANDOM_ACTION_PROBABILITY[i]/(ROUNDS)*(ROUNDS-j))
      PLAYER2.set_random_action_probability(RANDOM_ACTION_PROBABILITY[i]/(ROUNDS)*(ROUNDS-j))
      training_results, training_reward_values, training_board_spread = play_games(PLAYER1, PLAYER2, TRAINING_EPISODES, ACTION_LIMIT, WOOL, BRICK, LUMBER, GRAIN, ORE,ROUNDS,j)
      all_training_results.extend(training_results)
      all_training_reward_values.append(training_reward_values)
      all_training_board_spread.append(training_board_spread)
      PLAYER1.print_transition_information(PLAYER1.get_transitions_information())
      visited_states.append(PLAYER1.get_transitions_information()[1])
      PLAYER1.save_transition_information()

      # Games in validation are played
      PLAYER1.set_random_action_probability(0)
      PLAYER2.set_random_action_probability(0)
      optional_results, optional_reward_values, optional_board_spread = play_games(PLAYER2, PLAYER3, VALIDATION_EPISODES, ACTION_LIMIT, WOOL, BRICK, LUMBER, GRAIN, ORE,ROUNDS,j)
      validation_results, validation_reward_values, validation_board_spread = play_games(PLAYER1, PLAYER3, VALIDATION_EPISODES, ACTION_LIMIT, WOOL, BRICK, LUMBER, GRAIN, ORE,ROUNDS,j)
      all_validation_results.extend(validation_results)
      all_validation_reward_values.append(validation_reward_values)
      all_validation_board_spread.append(validation_board_spread)
      print("Round " + str(j+1) + " completed!")
      print("")
      


    # Plots are generated
    plot_heatmap(all_training_board_spread,all_validation_board_spread,ROUNDS,TRAINING_EPISODES, VALIDATION_EPISODES, LEARNING_RATE,DISCOUNT_RATE)
    plot_results(all_training_results, ROUNDS, TRAINING_EPISODES, LEARNING_RATE, DISCOUNT_RATE, "Training Information")
    plot_results(all_validation_results, ROUNDS, VALIDATION_EPISODES, LEARNING_RATE, DISCOUNT_RATE, "Validation Information")

    plot_comparison(all_training_reward_values, all_validation_reward_values, visited_states, ROUNDS, (TRAINING_EPISODES+VALIDATION_EPISODES), LEARNING_RATE, DISCOUNT_RATE)
    plot_comparison_2(all_training_reward_values, all_validation_reward_values, all_training_results, all_validation_results, ROUNDS, (TRAINING_EPISODES+VALIDATION_EPISODES), LEARNING_RATE, DISCOUNT_RATE)
    plot_comparison_3(all_training_results, all_validation_results, visited_states, ROUNDS, (TRAINING_EPISODES+VALIDATION_EPISODES), LEARNING_RATE, DISCOUNT_RATE)
    display_results(all_training_results, all_validation_results, ROUNDS, (TRAINING_EPISODES+VALIDATION_EPISODES), LEARNING_RATE, DISCOUNT_RATE)
    print("")

    # In case sensitivity analysis is performed
    # sensitivity_training_results.append(all_training_results)
    # sensitivity_validation_results.append(all_validation_results)
    
# plot_sensitivity(sensitivity_training_results, ROUNDS, TRAINING_EPISODES, len(RANDOM_ACTION_PROBABILITY), LEARNING_RATE, DISCOUNT_RATE, RANDOM_ACTION_PROBABILITY)
# plot_sensitivity(sensitivity_validation_results, ROUNDS, VALIDATION_EPISODES, len(RANDOM_ACTION_PROBABILITY), LEARNING_RATE, DISCOUNT_RATE, RANDOM_ACTION_PROBABILITY)

