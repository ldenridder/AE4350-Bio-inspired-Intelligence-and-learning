'''
Created on September 17, 2021
@author: Luc den Ridder
'''

from Board_Environment import Board_Environment
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

sns.set_theme()
save_folder = "B:/Documenten/Master/Bio-Inspired/New Method/plots"

def play_games(player1, player2, num_games, action_limit, wool, brick, lumber, grain, ore, rounds,round):
  """
  Plays a specified amount of games of Catan between player1, who goes first,
  and player2, who goes second.  The games will be stopped after the given limit on actions.
  This function outputs an array of arrays formatted as followed (only showing game 1's info):
  
  PRECONDITIONS:
  1)Both player1 and player2 inherit the Player class
  2)Both player1 and player2 play legal actions only
  """
  
  environment = Board_Environment()
  player1.set_environment(environment)
  player2.set_environment(environment)  

  
  game_results = [[-1,-1,-1,-1,-1,-1] for j in range(num_games)] 
  reward_value_results = []
  board_spread = []
  for j in range(num_games):
      
      action_counter = 0
      reward_values = 0
      # Active player is switched after each round
      if j % 2 == 0:
          active_player = player1
          environment.player_turn = True
      else:
          active_player = player2
          environment.player_turn = False
          
      # The game is played within the while loop
      while not environment.check_terminal(player1,player2) and action_counter < action_limit:
          environment.resource_update(player1,player2)
          reward_value = 0
          try:
              next_action, reward_value = active_player.get_next_action(player1,player2)
              environment.make_action(next_action,player1,player2)
          except:
              pass
          
          action_counter = action_counter + 1
          reward_values += reward_value
          
          # Active player is switched every turn
          if active_player is player1:
              active_player = player2
          else:
              active_player = player1
          environment.player_turn = not environment.player_turn 

      else:
          # Determine game result
          if Counter(environment.spots)[1] + 2*Counter(environment.spots)[3] == 10:
                game_results[j][0] = 0
          elif  Counter(environment.spots)[2] + 2*Counter(environment.spots)[4] == 10:
                game_results[j][0] = 1
          else:
                game_results[j][0] = 2
              
          game_results[j][1] = action_counter
          game_results[j][2] = Counter(environment.spots)[1]
          game_results[j][3] = Counter(environment.spots)[2]
          game_results[j][4] = Counter(environment.spots)[3]
          game_results[j][5] = Counter(environment.spots)[4]

          reward_value_results.append(reward_values)
          board_spread.append(environment.spots)
          
          # Print extra information for last couple of rounds
          if rounds-round < 5 and (j == num_games-1 or j == num_games-2):
                  environment.map_update(rounds,round,j)
                  print(f'P1 Wool = {player1.wool}')
                  print(f'P1 Brick = {player1.brick}')
                  print(f'P1 Lumber = {player1.lumber}')
                  print(f'P1 Grain = {player1.grain}')
                  print(f'P1 Ore = {player1.ore}')
                  print(f'P2 Wool = {player2.wool}')
                  print(f'P2 Brick = {player2.brick}')
                  print(f'P2 Lumber = {player2.lumber}')
                  print(f'P2 Grain = {player2.grain}')
                  print(f'P2 Ore = {player2.ore}')
                  print(f'Action counter = {action_counter}')
          player1.game_completed()
          player2.game_completed()
          environment.reset_environment()
          player1.reset_player(wool, brick, lumber, grain, ore)
          player2.reset_player(wool, brick, lumber, grain, ore)        
   
  return game_results, reward_value_results, board_spread

def display_results(game_results, game_results_validation, rounds, interval, LEARNING_RATE, DISCOUNT_RATE):
  """
  Prints the results of a round in a easy to understand format.  
  """


  game_wins = [0,0,0,0]
  total_actions = 0
  max_actions_made = float("-inf")
  min_actions_made = float("inf")
  for result in game_results:
      total_actions = total_actions + result[1]
      if result[1] < min_actions_made:
          min_actions_made = result[1]
      if result[1] > max_actions_made:
          max_actions_made = result[1]      
      game_wins[result[0]] = game_wins[result[0]] + 1

  game_wins_val = [0,0,0,0]
  total_actions = 0
  max_actions_made = float("-inf")
  min_actions_made = float("inf")
  for result in game_results_validation:
      total_actions = total_actions + result[1]
      if result[1] < min_actions_made:
          min_actions_made = result[1]
      if result[1] > max_actions_made:
          max_actions_made = result[1]      
      game_wins_val[result[0]] = game_wins_val[result[0]] + 1
  

  print("Games Played: ".ljust(35), len(game_results))
  print("Player 1 wins: ".ljust(35), game_wins[0])
  print("Player 2 wins: ".ljust(35), game_wins[1])
  print("Games exceeded action limit: ".ljust(35), game_wins[2])

  print("Total actions made: ".ljust(35), total_actions)  
  print("Average actions made: ".ljust(35), total_actions/len(game_results))
  print("Max actions made: ".ljust(35), max_actions_made)
  print("Min actions made: ".ljust(35), min_actions_made)
  
  all_game_wins = pd.DataFrame({'Names': ['Player 1','Player 2','Exceeds action limit','Tie'],
                                'Training': game_wins, 'Validation': game_wins_val})

  print(all_game_wins.set_index('Names').T)
  plt.figure()
  all_game_wins.set_index('Names').T.plot(kind='bar', stacked=True)
  plt.ylabel('Amount of wins')
  plt.grid(axis='x')
  plt.tight_layout()
  plt.savefig(os.path.join(save_folder,f"Display-{rounds}-{interval}_LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
  plt.close()

def plot_results(all_game_results, rounds, interval, LEARNING_RATE, DISCOUNT_RATE, title="End of Game Results"):
  """
  Plot the results of all rounds
  """
  player1_wins = [0 for _ in range(int(len(all_game_results)/interval))]
  player2_wins = [0 for _ in range(int(len(all_game_results)/interval))]
  action_limit = [0 for _ in range(int(len(all_game_results)/interval))]
  
  player1_wins_moving = []
  player2_wins_moving = []
  action_limit_moving = []
  smoothing_factor = 0.5

  for j in range(int(len(all_game_results)/interval)):
      for i in range(interval):
          if all_game_results[j*interval + i][0] == 0:
              player1_wins[j] = player1_wins[j] + 1
          elif all_game_results[j*interval + i][0] == 1:
              player2_wins[j] = player2_wins[j] + 1
          else:
              action_limit[j] = action_limit[j] + 1


  player1_wins_moving.append(player1_wins[0])
  player2_wins_moving.append(player2_wins[0])
  action_limit_moving.append(action_limit[0])
  i = 1

  # Create a function that plots a moving average
  while i < len(player1_wins):
    player1_window_average = round((smoothing_factor*player1_wins[i])+ (1-smoothing_factor)*player1_wins_moving[-1], 2)
    player1_wins_moving.append(player1_window_average)
    player2_window_average = round((smoothing_factor*player2_wins[i])+ (1-smoothing_factor)*player2_wins_moving[-1], 2)
    player2_wins_moving.append(player2_window_average)
    action_limit_window_average = round((smoothing_factor*action_limit[i])+ (1-smoothing_factor)*action_limit_moving[-1], 2)
    action_limit_moving.append(action_limit_window_average)
    i += 1        

  plt.figure()
  p1_win_graph, = plt.plot(player1_wins, label = "First player wins",color='tab:green')
  p2_win_graph, = plt.plot(player2_wins, label = "Second player wins",color='tab:red')
  action_limit_graph, = plt.plot(action_limit, label = "Action limit reached",color='tab:blue')
  plt.plot(player1_wins_moving,color='tab:green',linestyle='--', alpha=0.5)
  plt.plot(player2_wins_moving,color='tab:red',linestyle='--', alpha=0.5)
  plt.plot(action_limit_moving,color='tab:blue',linestyle='--', alpha=0.5)
  plt.ylabel("Occurance per " + str(interval) + " games")
  plt.xlabel("Interval")
  plt.title(f"Learning rate = {LEARNING_RATE} & Discount rate = {DISCOUNT_RATE}")
  plt.legend(handles=[p1_win_graph, p2_win_graph, action_limit_graph])
  plt.tight_layout()
  plt.savefig(os.path.join(save_folder,f"Wins_IN-{rounds}-{interval}_LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
  plt.close()

def plot_comparison(training_reward_values, validation_reward_values, visited_states, rounds, interval, LEARNING_RATE, DISCOUNT_RATE):
  """
  Plots average total reward values per round and the toal number of states visitied by player 1
  """
  mean_training_reward_values = [np.mean(training_reward_values[i]) for i in range(np.shape(training_reward_values)[0])]
  mean_validation_reward_values = [np.mean(validation_reward_values[i]) for i in range(np.shape(validation_reward_values)[0])]

  plt.figure()
  fig, ax1 = plt.subplots()
  training_graph, = ax1.plot(mean_training_reward_values, label = "Training", color='tab:green')
  validation_graph, = ax1.plot(mean_validation_reward_values, label = "Validation" , color='tab:red')

  ax2 = ax1.twinx() 
  states_graph, = ax2.plot(visited_states, label = "Visited States", color='tab:blue')

  ax1.set_ylabel("Average Total Reward Values per round")
  ax1.set_ylim(bottom=0)
  ax2.set_ylabel("Total Number of States Visited by player")
  ax2.set_ylim(bottom=0)
  ax1.set_xlabel("Interval")

  plt.legend(handles=[training_graph, validation_graph, states_graph])
  plt.tight_layout()
  plt.savefig(os.path.join(save_folder,f"Reward_values-{rounds}-{interval}_LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
  plt.close()

def plot_comparison_2(training_reward_values, validation_reward_values, training_results, validation_results, rounds, interval, LEARNING_RATE, DISCOUNT_RATE):
  """
  Plots average total reward values per round and average number of total actions per round
  """
  mean_training_reward_values = [np.mean(training_reward_values[i]) for i in range(np.shape(training_reward_values)[0])]
  mean_validation_reward_values = [np.mean(validation_reward_values[i]) for i in range(np.shape(validation_reward_values)[0])]

  j = -1
  training_actions = [[] for i in range(rounds)]
  for i in range(np.shape(training_results)[0]):
      if i%int(np.shape(training_results)[0]/rounds) == 0:
          j += 1
      training_actions[j].append(training_results[i][1])

  j = -1
  validation_actions = [[] for i in range(rounds)]
  for i in range(np.shape(validation_results)[0]):
      if i%int(np.shape(validation_results)[0]/rounds) == 0:
          j += 1
      validation_actions[j].append(validation_results[i][1])
    
  sum_training_actions = [np.mean(training_actions[i]) for i in range(rounds)]
  sum_validation_actions = [np.mean(validation_actions[i]) for i in range(rounds)]  

  plt.figure()
  fig, ax1 = plt.subplots()
  training_graph, = ax1.plot(mean_training_reward_values, label = "Training Reward Values", color='tab:green')
  validation_graph, = ax1.plot(mean_validation_reward_values, label = "Validation Reward Values" , color='tab:red')

  ax2 = ax1.twinx() 
  training_actions_graph, = ax2.plot(sum_training_actions, label = "Training Total Actions", color='tab:green',linestyle='--')
  validation_actions_graph, = ax2.plot(sum_validation_actions, label = "Validation Total Actions" , color='tab:red',linestyle='--')

  ax1.set_ylabel("Average Total Reward Values per round")
  ax1.set_ylim(bottom=0)
  ax2.set_ylim(bottom=0)
  ax2.set_ylabel("Average Number of Total Actions per round")
  ax1.set_xlabel("Interval")
  #ax1.title(f"Learning rate = {LEARNING_RATE} & Discount rate = {DISCOUNT_RATE}")
  plt.legend(handles=[training_graph, validation_graph, training_actions_graph,validation_actions_graph])
  plt.tight_layout()
  plt.savefig(os.path.join(save_folder,f"Reward_values_2-{rounds}-{interval}_LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
  plt.close()

def plot_comparison_3(training_results, validation_results, visited_states, rounds, interval, LEARNING_RATE, DISCOUNT_RATE):
  """
  Plots average number of total actionss per round and the toal number of states visitied by player 1
  """
  j = -1
  training_actions = [[] for i in range(rounds)]
  for i in range(np.shape(training_results)[0]):
      if i%int(np.shape(training_results)[0]/rounds) == 0:
          j += 1
      training_actions[j].append(training_results[i][1])

  j = -1
  validation_actions = [[] for i in range(rounds)]
  for i in range(np.shape(validation_results)[0]):
      if i%int(np.shape(validation_results)[0]/rounds) == 0:
          j += 1
      validation_actions[j].append(validation_results[i][1])
    
  sum_training_actions = [np.mean(training_actions[i]) for i in range(rounds)]
  sum_validation_actions = [np.mean(validation_actions[i]) for i in range(rounds)]  

  plt.figure()
  fig, ax1 = plt.subplots()

  ax2 = ax1.twinx() 
  training_actions_graph, = ax1.plot(sum_training_actions, label = "Training Total Actions", color='tab:green')
  validation_actions_graph, = ax1.plot(sum_validation_actions, label = "Validation Total Actions" , color='tab:red')
  states_graph, = ax2.plot(visited_states, label = "Visited States", color='tab:blue')

  ax1.set_ylabel("Average Number of Total Actions per round")
  ax1.set_ylim(bottom=0)
  ax2.set_ylabel("Total Number of States Visited by player")
  ax2.set_ylim(bottom=0)
  ax1.set_xlabel("Interval")
  plt.legend(handles=[training_actions_graph,validation_actions_graph, states_graph])

  plt.tight_layout()
  plt.savefig(os.path.join(save_folder,f"Reward_values_3-{rounds}-{interval}_LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
  plt.close()

def plot_sensitivity(all_results, rounds, episodes, interval, LEARNING_RATE, DISCOUNT_RATE, RANDOM_ACTION_PROBABILITY, title="End of Game Results"):
  """
  Plot the sensitivity plots of each round
  """

  player1_wins = [0 for _ in range(interval)]
  action_limit = [0 for _ in range(interval)]


  for k in range(interval):
      player1_wins[k] = [0 for _ in range(int(len(all_results[k])/episodes))]
      action_limit[k] = [0 for _ in range(int(len(all_results[k])/episodes))]
      
      for j in range(int(len(all_results[k])/episodes)):
          for i in range(episodes):
              if all_results[k][j*episodes + i][0] == 0:
                  player1_wins[k][j] = player1_wins[k][j] + 1
              elif all_results[k][j*episodes + i][0] == 2:
                  action_limit[k][j] = action_limit[k][j] + 1
          
  
  # In this case the RAP is plotted
  plt.figure()
  p1_win_R0_graph, = plt.plot(player1_wins[0], label = f"Random action probability: {RANDOM_ACTION_PROBABILITY[0]}")
  p1_win_R1_graph, = plt.plot(player1_wins[1], label = f"Random action probability: {RANDOM_ACTION_PROBABILITY[1]}")
  p1_win_R2_graph, = plt.plot(player1_wins[2], label = f"Random action probability: {RANDOM_ACTION_PROBABILITY[2]}")
  # p1_win_R3_graph, = plt.plot(player1_wins[3], label = "Reward Values: Option 4")
  # p1_win_R4_graph, = plt.plot(player1_wins[4], label = "Reward Values: Option 5")
  plt.ylabel("Occurance per " + str(episodes) + " games")
  plt.xlabel("Interval")
  plt.title(f"Player 1 wins & Learning rate = {LEARNING_RATE} & Discount rate = {DISCOUNT_RATE}")
  plt.legend(handles=[p1_win_R0_graph, p1_win_R1_graph, p1_win_R2_graph])#,p1_win_R3_graph,p1_win_R4_graph])
  plt.tight_layout()
  plt.savefig(os.path.join(save_folder,f"P1Win_{rounds}-{episodes}-{interval}__LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
  plt.close()
  
  plt.figure()
  action_limit_R0_graph, = plt.plot(action_limit[0], label = f"Random action probability: {RANDOM_ACTION_PROBABILITY[0]}")
  action_limit_R1_graph, = plt.plot(action_limit[1], label = f"Random action probability: {RANDOM_ACTION_PROBABILITY[1]}")
  action_limit_R2_graph, = plt.plot(action_limit[2], label = f"Random action probability: {RANDOM_ACTION_PROBABILITY[2]}")
  # action_limit_R3_graph, = plt.plot(action_limit[3], label = "Reward Values: Option 4")
  # action_limit_R4_graph, = plt.plot(action_limit[4], label = "Reward Values: Option 5")
  plt.ylabel("Occurance per " + str(episodes) + " games")
  plt.xlabel("Interval")
  plt.title(f"Action Limit & Learning rate = {LEARNING_RATE} & Discount rate = {DISCOUNT_RATE}")
  plt.legend(handles=[action_limit_R0_graph, action_limit_R1_graph, action_limit_R2_graph])#, action_limit_R3_graph, action_limit_R4_graph])
  plt.tight_layout()
  plt.savefig(os.path.join(save_folder,f"AL_{rounds}-{episodes}-{interval}_LR-{LEARNING_RATE}-DR-{DISCOUNT_RATE}.png"))
  plt.close()

def plot_heatmap(all_training_board_spread,all_validation_board_spread,ROUNDS,TRAINING_EPISODES, VALIDATION_EPISODES, LEARNING_RATE,DISCOUNT_RATE):
    """
    Plots the heatmaps of player 1 and 2 during training and validation
    """

    all_training_board_spread = np.reshape(all_training_board_spread, (-1,ROUNDS*TRAINING_EPISODES))
    all_validation_board_spread = np.reshape(all_validation_board_spread, (-1,ROUNDS*VALIDATION_EPISODES))


    training_board = {'P1_S':[],'P2_S':[],'P1_C':[],'P2_C':[]}
    validation_board = {'P1_S':[],'P2_S':[],'P1_C':[],'P2_C':[]}

    # Counts the number of settlements and cities on each of the spots
    for i in range(np.shape(all_training_board_spread)[0]):
      training_board['P1_S'].append(int((list(all_training_board_spread[i]).count(1)/1)**4))
      training_board['P2_S'].append(int((list(all_training_board_spread[i]).count(2)/1)**4))
      training_board['P1_C'].append(int((list(all_training_board_spread[i]).count(3)/1)**4))
      training_board['P2_C'].append(int((list(all_training_board_spread[i]).count(4)/1)**4))
      validation_board['P1_S'].append(int((list(all_validation_board_spread[i]).count(1)/1)**4))
      validation_board['P2_S'].append(int((list(all_validation_board_spread[i]).count(2)/1)**4))
      validation_board['P1_C'].append(int((list(all_validation_board_spread[i]).count(3)/1)**4))
      validation_board['P2_C'].append(int((list(all_validation_board_spread[i]).count(4)/1)**4))


    environment = Board_Environment()

    # Organise the lists in a Dataframe
    all_training_boards_p1 = pd.DataFrame({
      'X-Loc': np.array([environment.xloc for i in range(2)]).flatten(),
      'Y-Loc': np.array([environment.yloc for i in range(2)]).flatten(),
      'Size': np.array([training_board['P1_S'],training_board['P1_C']]).flatten(),
      'Type': np.array([['Settlement']*54,['City']*54]).flatten()})

    all_training_boards_p2 = pd.DataFrame({
      'X-Loc': np.array([environment.xloc for i in range(2)]).flatten(),
      'Y-Loc': np.array([environment.yloc for i in range(2)]).flatten(),
      'Size': np.array([training_board['P2_S'],training_board['P2_C']]).flatten(),
      'Type': np.array([['Settlement']*54,['City']*54]).flatten()})

    all_validation_boards_p1 = pd.DataFrame({
      'X-Loc': np.array([environment.xloc for i in range(2)]).flatten(),
      'Y-Loc': np.array([environment.yloc for i in range(2)]).flatten(),
      'Size': np.array([validation_board['P1_S'],validation_board['P1_C']]).flatten(),
      'Type': np.array([['Settlement']*54,['City']*54]).flatten()})

    all_validation_boards_p2 = pd.DataFrame({
      'X-Loc': np.array([environment.xloc for i in range(2)]).flatten(),
      'Y-Loc': np.array([environment.yloc for i in range(2)]).flatten(),
      'Size': np.array([validation_board['P2_S'],validation_board['P2_C']]).flatten(),
      'Type': np.array([['Settlement']*54,['City']*54]).flatten()})

    # Plots each of the dataframes on the board
    plt.figure()
    img = plt.imread("board.jpg")
    plt.imshow(img, zorder=0, extent=[-5, 5, -8, 8],alpha=0.5)
    sns.scatterplot(data=all_training_boards_p1, x='X-Loc',y='Y-Loc', hue='Type', size='Size',sizes = (0,400), alpha = 0.7, palette=['tab:red','tab:blue'])
    plt.grid(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels[:3], handles[:3]))
    plt.legend(by_label.values(), by_label.keys(),loc=0,bbox_to_anchor =(1.0, 1.0))       
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder,f"Heatmap_Training_P1-{ROUNDS}-{TRAINING_EPISODES}_LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
    plt.close()

    plt.figure()
    img = plt.imread("board.jpg")
    plt.imshow(img, zorder=0, extent=[-5, 5, -8, 8],alpha=0.5)
    sns.scatterplot(data=all_training_boards_p2, x='X-Loc',y='Y-Loc', hue='Type', size='Size', sizes = (0,400), alpha = 0.7, palette=['tab:red','tab:blue'])
    plt.grid(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels[:3], handles[:3]))
    plt.legend(by_label.values(), by_label.keys(),loc=0,bbox_to_anchor =(1.0, 1.0))       
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder,f"Heatmap_Training_P2-{ROUNDS}-{TRAINING_EPISODES}_LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
    plt.close()

    plt.figure()
    img = plt.imread("board.jpg")
    plt.imshow(img, zorder=0, extent=[-5, 5, -8, 8],alpha=0.5)
    sns.scatterplot(data=all_validation_boards_p1, x='X-Loc',y='Y-Loc', hue='Type', size='Size', sizes = (0,400), alpha = 0.7, palette=['tab:red','tab:blue'])
    plt.grid(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels[:3], handles[:3]))
    plt.legend(by_label.values(), by_label.keys(),loc=0,bbox_to_anchor =(1.0, 1.0))        
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder,f"Heatmap_Validation_P1-{ROUNDS}-{VALIDATION_EPISODES}_LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
    plt.close()

    plt.figure()
    img = plt.imread("board.jpg")
    plt.imshow(img, zorder=0, extent=[-5, 5, -8, 8],alpha=0.5)
    sns.scatterplot(data=all_validation_boards_p2, x='X-Loc',y='Y-Loc', hue='Type', size='Size', sizes = (0,400), alpha = 0.7, palette=['tab:red','tab:blue'])
    plt.grid(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels[:3], handles[:3]))
    plt.legend(by_label.values(), by_label.keys(),loc=0,bbox_to_anchor =(1.0, 1.0))        
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder,f"Heatmap_Validation_P2-{ROUNDS}-{VALIDATION_EPISODES}_LR-{LEARNING_RATE}_DR-{DISCOUNT_RATE}.png"))
    plt.close()