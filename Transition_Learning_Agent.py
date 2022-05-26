'''
Created on September 17, 2021
@author: Luc den Ridder
'''

import random
import json
import numpy as np
from ast import literal_eval

class Transition_Learning_Agent:
  """
  Learning agent based on a RL method that values the transition between states
  """

  def __init__(self, the_learning_rate, the_discount_rate, the_wool, the_brick, the_lumber, the_grain, the_ore, the_random_action_probability, the_reward_value_1, the_reward_value_2, the_reward_value_3, the_reward_value_4, the_reward_value_5, info_location=None, the_environment=None):
      """
      Initialize the learning agent 
      """
      self.random_action_probability = the_random_action_probability
      self.alpha = the_learning_rate    
      self.gamma = the_discount_rate
      self.environment = the_environment
      self.reward_value_1 = the_reward_value_1
      self.reward_value_2 = the_reward_value_2
      self.reward_value_3 = the_reward_value_3
      self.reward_value_4 = the_reward_value_4
      self.reward_value_5 = the_reward_value_5 
      self.state_t0 = None
      self.state_t1 = None 
      
      self.wool = the_wool
      self.brick = the_brick
      self.lumber = the_lumber
      self.grain = the_grain
      self.ore = the_ore
      
      if not info_location is None:
          self.load_transition_information(info_location)
      else:
          self.T = {}

  def reset_player(self, the_wool, the_brick, the_lumber, the_grain, the_ore):
      """
      Resets the resource of the player
      """
      
      self.wool = the_wool
      self.brick = the_brick
      self.lumber = the_lumber
      self.grain = the_grain
      self.ore = the_ore      
    
  def set_random_action_probability(self, probability):
      """
      Sets the random action probability
      """
      self.random_action_probability = probability

  def set_learning_rate(self, the_learning_rate):
      """
      Sets the learning rate
      """
      self.alpha = the_learning_rate

  def set_environment(self, the_environment):
      """
      Sets the Board Environment as known by the learning agent
      """
      self.environment = the_environment
  
  def game_completed(self):
      """
      Updates the transition values after the game is completed and before the board is cleared.
      """
      transition_states = (self.state_t0 ,self.state_t1)

      self.T[transition_states] = self.T[transition_states] + self.alpha * self.reward_function(self.state_t0,self.state_t1)
      self.state_t0 = None
      self.state_t1 = None
      
  def get_states(self, spots, roads):
      """
      Computes the states based on the spots and roads from the environment or from a potential environment
      and returns the states as tuples with the format:
      [(own_settlements, own_cities, own_positional_value, own_resource_diversity, own_roads), ...]      
      """
      states = [[0,0,0,0,0] for j in range(len(spots))] 
      
    #   print(f'roads: {np.shape(roads[0])}{roads}')
    #   print(f'spots: {np.shape(spots[0])}{spots}')
      for j in range(len(spots)):       
            resource_diversity = []
            for i in range(len(spots[j])):
                if spots[j][i] == 1 and self.environment.player_turn == True:
                    states[j][0] = states[j][0] + 1
                    states[j][2] =  states[j][2] + sum([7-abs(k-7) for k in self.environment.value[i]])/10
                    for k in self.environment.resource[i]:                       
                        if k not in resource_diversity:
                            resource_diversity.append(k)
                            states[j][3] = states[j][3] + len(resource_diversity)
                if spots[j][i] == 3 and self.environment.player_turn == True:
                    states[j][1] = states[j][1] + 1
                    states[j][2] =  states[j][2] + sum([7-abs(k-7) for k in self.environment.value[i]])/5
                    for k in self.environment.resource[i]:                       
                        if k not in resource_diversity:
                            resource_diversity.append(k)
                            states[j][3] = states[j][3] + len(resource_diversity)
                if spots[j][i] == 2 and self.environment.player_turn == False:
                    states[j][0] = states[j][0] + 1
                    states[j][2] =  states[j][2] + sum([7-abs(k-7) for k in self.environment.value[i]])/10
                    for k in self.environment.resource[i]:                       
                        if k not in resource_diversity:
                            resource_diversity.append(k)
                            states[j][3] = states[j][3] + len(resource_diversity)
                if spots[j][i] == 4 and self.environment.player_turn == False:
                    states[j][1] = states[j][1] + 1
                    states[j][2] =  states[j][2] + sum([7-abs(k-7) for k in self.environment.value[i]])/5
                    for k in self.environment.resource[i]:                       
                        if k not in resource_diversity:
                            resource_diversity.append(k)
                            states[j][3] = states[j][3] + len(resource_diversity)

            
            for i in range(len(roads[j])):
                for k in range(len(roads[j][i])):
                    if roads[j][i][k] == 1 and self.environment.player_turn == True:
                        states[j][4] =  states[j][4] + 1
                    if roads[j][i][k] == 2 and self.environment.player_turn == False:
                        states[j][4] =  states[j][4] + 1

      return [tuple(i) for i in states]
                               
  def get_transition_policy(self, possible_states, initial_transition_value=10):
      """
      Gets the desired transition from the current state to a following possible state based on a highest transition value
      If the possible transition does not exist yet, it will create it.
      """
      state_t2 = tuple(self.get_states([self.environment.spots],[self.environment.roads])[0])
      explored_transitions = {}
      for i in possible_states:
          if explored_transitions.get((state_t2, tuple(i))) is None:
              if self.T.get((state_t2, tuple(i))) is None:
                  self.T.update({(state_t2, tuple(i)):initial_transition_value})
              explored_transitions.update({(state_t2, tuple(i)):self.T.get((state_t2, tuple(i)))})
         
      if random != 0 and random.random() < self.random_action_probability:
          try:
               return list(explored_transitions.keys())[random.randint(0, len(explored_transitions)-1)]
          except:   
              return []
      else:
          try:
              reverse_dict = {j:i for i,j in explored_transitions.items()}
              return reverse_dict.get(max(reverse_dict))   
          except:
              return []    
         
  def get_max_transition_value(self):
      """
      Look ahead a given number of actions and return the maximal value associated 
      with a action of that depth.         
      """

      possible_transition_values = []
      state_t2 = self.get_states([self.environment.spots],[self.environment.roads])[0]
      for states,value in self.T.items():

          if value > float("-inf") and states[0] == state_t2:

              possible_transition_values.append(value)
      if possible_transition_values == []:
          return None

      return max(possible_transition_values)

  def get_next_action(self,player1,player2):
      """
      Gets the next action based on the current state and environment by looking at the transition values. Only legitame actions can be made
      """
      if self.state_t0 is not None:
          transition_states = (self.state_t0 ,self.state_t1) 
          reward_value = self.reward_function(self.state_t0,self.state_t1)  
          try:
              if self.get_max_transition_value() == None:
                  self.T[transition_states] = self.T[transition_states] + self.alpha * reward_value
              else:                  
                  self.T[transition_states] = self.T[transition_states] + self.alpha * (reward_value + self.gamma*self.get_max_transition_value() - self.T[transition_states])
          except:
              self.T[transition_states] = self.T[transition_states] + self.alpha * reward_value

      
      self.state_t0 = self.get_states([self.environment.spots],[self.environment.roads])[0]
      possible_actions = self.environment.get_possible_actions(player1,player2)

      if possible_actions == [[]]:
          possible_next_states = [self.state_t0]
      else:    
          possible_spots, possible_edges = self.environment.get_possible_spots_and_edges(possible_actions,player1,player2)
          possible_next_states = self.get_states(possible_spots,possible_edges)

      self.state_t1 = self.get_transition_policy(possible_next_states)[1]   
      desired_action = []
      if possible_actions != [[]]:
          for j in range(len(possible_next_states)):
              if tuple(possible_next_states[j]) == self.state_t1:
                  desired_action.append(possible_actions[j])
          return desired_action[random.randint(0,len(desired_action)-1)], reward_value
      else:
          return desired_action, reward_value

  def reward_function(self,state_t0, state_t1):
       """
       Reward for transitioning from state_t0 to state_t1, positional value is rewarded more while settlements and resource diversity rewarded less 
       """   
       
       return (state_t1[0]-state_t0[0])*self.reward_value_1 + (state_t1[1]-state_t0[1])*self.reward_value_2 + (state_t1[2]-state_t0[2])*self.reward_value_3 + (state_t1[3]-state_t0[3])*self.reward_value_4 + (state_t1[4]-state_t0[4])*self.reward_value_5 

  def get_transitions_information(self):
      """
      Get an array of of information about the dictionary self.T .
      It returns the information in the form:
      [num_transitions, num_start_of_transitions, avg_value, max_value, min_value]
      """
      start_of_transitions = {}
      max_value = float("-inf")
      min_value = float("inf")
      total_value = 0
      
      for states,value in self.T.items():
          if start_of_transitions.get(states[0]) is None:
              start_of_transitions.update({states[0]:0})
          if value > max_value:
              max_value = value
          if value < min_value:
              min_value = value
          total_value = total_value + value
          
      return [len(self.T), len(start_of_transitions), float(total_value/len(self.T)), max_value, min_value]
    
  def print_transition_information(self, info):
      """
      Prints the output of get_transitions_information in a easy to understand format.
      """
      print("Total number of transitions: ".ljust(35), info[0])        
      print("Total number of visited states: ".ljust(35), info[1])
      print("Average value for transition: ".ljust(35), info[2])
      print("Maximum value for transition: ".ljust(35), info[3])
      print("Minimum value for transition: ".ljust(35), info[4])
      
  def save_transition_information(self, file_name="data.json"):
      """
      Saves the current transitions information to a specified
      json file. 
      """
      with open(file_name, 'w') as fp:
          json.dump({str(k): v for k,v in self.T.items()}, fp)
           
  def load_transition_information(self, file_name):
      """
      Loads transitions information from a desired json file.
      """
      with open(file_name, 'r') as fp:
          self.T = {literal_eval(k): v for k,v in json.load(fp).items()}