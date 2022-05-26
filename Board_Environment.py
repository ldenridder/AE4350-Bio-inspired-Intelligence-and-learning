'''
Created on August 28, 2021
@author: Luc den Ridder
'''

import copy
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
import os

save_folder = "B:/Documenten/Master/Bio-Inspired/New Method/plots2"

class Board_Environment:
    """
    A class to represent and play Catan
    """
    EMPTY_SPOT = 0
    P1_S = 1
    P2_S = 2
    P1_C = 3
    P2_C = 4
    MAX_INACTIVITY = 200
    HEIGHT = 52
    inactivity = 0
    
    
    def __init__(self, old_spots=None, the_player_turn=True):
        """
        Initializes a new instance of the Board class.  Unless specified otherwise, 
        the board will be created with a start board configuration.
        
        """    
        self.player_turn = the_player_turn 
        self.load_environment()
        
    
    def load_environment(self):
        """
        Initalizes the board based on the cards and generates the lists with grid locations and playable actions.
        Note that order for resource and values on the same row is equal, as is the case for neighbours and raods.
        """
        cards_value = [10,2,9,12,6,4,10,9,11,0,3,8,8,3,4,5,5,6,11]
        cards_resource = ['Ore','Wool','Lumber','Grain','Brick','Wool','Brick','Grain','Lumber','Nothing','Lumber','Ore','Lumber','Ore','Grain','Wool','Brick','Grain','Wool']
        cards_xloc = [-2,0,2,-3,-1,1,3,-4,-2,0,2,4,-3,-1,1,3,-2,0,2]
        cards_yloc = [6,6,6,3,3,3,3,0,0,0,0,0,-3,-3,-3,-3,-6,-6,-6]

        self.resource = np.empty((11*17, 0)).tolist()
        self.value = np.empty((11*17, 0)).tolist()
        self.loc = []
           
        for i in range(len(cards_value)):
            j1 = 11*(cards_yloc[i]+10)+(cards_xloc[i]+5)
            j2 = 11*(cards_yloc[i]+9)+(cards_xloc[i]+6)
            j3 = 11*(cards_yloc[i]+7)+(cards_xloc[i]+6)   
            j4 = 11*(cards_yloc[i]+6)+(cards_xloc[i]+5)
            j5 = 11*(cards_yloc[i]+7)+(cards_xloc[i]+4)
            j6 = 11*(cards_yloc[i]+9)+(cards_xloc[i]+4)               
        
            self.resource[j1].append(cards_resource[i]) 
            self.resource[j2].append(cards_resource[i]) 
            self.resource[j3].append(cards_resource[i]) 
            self.resource[j4].append(cards_resource[i]) 
            self.resource[j5].append(cards_resource[i]) 
            self.resource[j6].append(cards_resource[i]) 

            self.value[j1].append(cards_value[i]) 
            self.value[j2].append(cards_value[i]) 
            self.value[j3].append(cards_value[i]) 
            self.value[j4].append(cards_value[i]) 
            self.value[j5].append(cards_value[i]) 
            self.value[j6].append(cards_value[i])     
        
        for i in range(len(self.resource)):
            if self.resource[i] != []:
                self.loc.append(i)
                
        self.resource = [x for x in self.resource if x != []]
        self.value = [x for x in self.value if x != []]

        self.spots = [0] * len(self.resource)
        self.xloc = [0] * len(self.resource)
        self.yloc = [0] * len(self.resource)
        self.neighbours = np.empty((len(self.resource),0)).tolist()
        self.roads = np.empty((len(self.resource),0)).tolist()
        
        for i, loc in enumerate(self.loc):
            self.xloc[i] = loc-int(loc/11)*11-5
            self.yloc[i] = int(loc/11)-8
        
        for i in range(len(self.resource)):
            for j in range(-2,3):
                for k in range(-1,2):
                    l = 11*(self.yloc[i]+8+j)+(self.xloc[i]+5+k)
                    
                    if l in self.loc and self.xloc[self.loc.index(l)] == self.xloc[i]+k \
                    and self.yloc[self.loc.index(l)] == self.yloc[i]+j and l != self.loc[i]:
                        
                        self.roads[i].append(0)
                        self.neighbours[i].append(self.loc.index(l))
                               
    def reset_environment(self):
        """
        Resets the current configuration of the game board to the original 
        starting position.
        """
        self.load_environment()
        self.player_turn = True
        self.inactivity = 0     
        
     
    def check_terminal(self,p1,p2):
        """
        Checks whether the game should be continued or ended
        """       
        if self.get_possible_actions(p1,p2) == []:
            self.inactivity+=1
            # print(self.inactivity)
        if (Counter(self.spots)[self.P1_S] + 2*Counter(self.spots)[self.P1_C] == 10) or (Counter(self.spots)[self.P2_S] + 2*Counter(self.spots)[self.P2_C] == 10):
            return True    
        
        if self.inactivity==self.MAX_INACTIVITY:

            return True
        return False
    
    def get_start_settlement_actions(self):
        """
        look for potential settlement locations at the start of the game
        """ 
        potential_locations = []
        for i in range(len(self.spots)):                          
            if self.spots[i] == self.EMPTY_SPOT: 
                neighbour_check = [self.spots[self.neighbours[i][j]] for j in range(len(self.neighbours[i]))]              
                if sum(neighbour_check) == self.EMPTY_SPOT:
                    potential_locations.append(i) 
    
        return potential_locations
    
    def get_settlement_actions(self):
        """
        look for potential settlement locations that borders a same-colored road at correct distance
        """         
        potential_locations = []
        for i in range(len(self.spots)):                          
            if self.spots[i] == self.EMPTY_SPOT: 
                neighbour_check = [self.spots[self.neighbours[i][j]] for j in range(len(self.neighbours[i]))]
                
                if sum(neighbour_check) == self.EMPTY_SPOT:
                    for j in range(len(self.roads[i])):
                        if (self.roads[i][j] == self.P1_S and self.player_turn == True) or (self.roads[i][j] == self.P2_S and self.player_turn == False):
                            potential_locations.append(i) 
                   
        return potential_locations
          
    def get_city_actions(self):
        """
        look for potential city locations that are a same-colored settlement
        """               
        potential_locations = []
        for i in range(len(self.spots)):
            if (self.spots[i] == self.P1_S and self.player_turn == True) or (self.spots[i] == self.P2_S and self.player_turn == False):
              potential_locations.append(i)  
                      
        return potential_locations
    
    def get_road_actions(self):
        """
        look for potential road locations that borders a same-colored road, settlement or city
        """       
        potential_locations = []
        for i in range(len(self.spots)):
            if ((self.spots[i] == self.P1_S or self.spots[i] == self.P1_C) and self.player_turn == True) \
            or ((self.spots[i] == self.P2_S or self.spots[i] == self.P2_C) and self.player_turn == False):
                for j in range(len(self.roads[i])):
                    if self.roads[i][j] == self.EMPTY_SPOT:
                        potential_locations.append([i,j])
            
            if self.spots[i] == self.EMPTY_SPOT and sum(self.roads[i]) > 0:
                if (self.P1_S in self.roads[i] and self.player_turn == True) or (self.P2_S in self.roads[i] and self.player_turn == False):
                    for j in range(len(self.roads[i])):
                        if self.roads[i][j] == self.EMPTY_SPOT:
                            potential_locations.append([i,j])

        return potential_locations
       
    def get_possible_actions(self,p1,p2):
        """
        Gets the possible actions that can be made from the current board configuration and player inventory.
        """
        start_check = 0
        for i in range(len(self.spots)):
            if ((self.spots[i] == self.P1_S or self.spots[i] == self.P1_C) and self.player_turn == True) or ((self.spots[i] == self.P2_S or self.spots[i] == self.P2_C) and self.player_turn == False):
                start_check += 1 
       
        if start_check <= 1:
            if (p1.wool > 0 and p1.brick > 0 and p1.lumber > 0 and p1.grain > 0 and self.player_turn == True) \
            or (p2.wool > 0 and p2.brick > 0 and p2.lumber > 0 and p2.grain > 0 and self.player_turn == False):
                possible_actions = self.get_start_settlement_actions()
        
        else:
            if (p1.wool > 0 and p1.brick > 0 and p1.lumber > 0 and p1.grain > 0 and self.player_turn == True) \
            or (p2.wool > 0 and p2.brick > 0 and p2.lumber > 0 and p2.grain > 0 and self.player_turn == False):
                possible_actions = self.get_settlement_actions()
            else: 
                possible_actions = []
                
            if (p1.ore > 2 and p1.grain > 1 and self.player_turn == True) \
            or (p2.ore > 2 and p2.grain > 1 and self.player_turn == False):    
                city_actions = self.get_city_actions()
                possible_actions.extend(city_actions)
                
            if (p1.brick > 0 and p1.lumber > 0 and self.player_turn == True) \
            or (p2.brick > 0 and p2.lumber > 0 and self.player_turn == False): 
                road_actions = self.get_road_actions()
                possible_actions.extend(road_actions)
        
            possible_actions.append([])

        return possible_actions
  
    def make_action(self, action, p1, p2, switch_player_turn=True, potential=False):
        """
        Makes a given action on the board, and (as long as is wanted) switches the indicator for
        which players turn it is.
        """
        if type(action) is list:
            if self.player_turn == True:
                self.roads[self.neighbours[action[0]][action[1]]][self.neighbours[self.neighbours[action[0]][action[1]]].index(action[0])] = self.P1_S
                self.roads[action[0]][action[1]] = self.P1_S
                if potential==False:
                    p1.brick -= 1
                    p1.lumber -= 1
            elif self.player_turn == False:
                self.roads[self.neighbours[action[0]][action[1]]][self.neighbours[self.neighbours[action[0]][action[1]]].index(action[0])] = self.P2_S
                self.roads[action[0]][action[1]] = self.P2_S
                if potential==False:
                    p2.brick -= 1
                    p2.lumber -= 1
        else:
            if self.spots[action] == self.EMPTY_SPOT:
                if self.player_turn == True:
                    self.spots[action] = self.P1_S
                    if potential==False:
                        p1.wool -= 1
                        p1.brick -= 1
                        p1.lumber -= 1
                        p1.grain -= 1
                elif self.player_turn == False:
                    self.spots[action] = self.P2_S
                    if potential==False:
                        p2.wool -= 1
                        p2.brick -= 1
                        p2.lumber -= 1
                        p2.grain -= 1
            elif self.spots[action] == self.P1_S and self.player_turn == True:
                self.spots[action] = self.P1_C
                if potential==False:
                    p1.ore -= 3
                    p1.grain -= 2
            elif self.spots[action] == self.P2_S and self.player_turn == False:
                self.spots[action] = self.P2_C
                if potential==False:
                    p2.ore -= 3
                    p2.grain -= 2  

    def get_possible_spots_and_edges(self, actions, p1, p2):
        """
        Get's the potential spots for the board if it makes any of the given actions.
        If actions is None then returns it's own current spots.
        """
        if actions is None:
            return self.spots, self.roads
        
        answer_spots = []
        answer_roads = []

        for action in actions:
            if action == []:
                answer_spots.append(self.spots)
                answer_roads.append(self.roads)
            else:
                original_spots = copy.deepcopy(self.spots)
                original_roads = copy.deepcopy(self.roads)
                self.make_action(action, p1, p2, switch_player_turn=False, potential=True)
                answer_spots.append(self.spots)
                answer_roads.append(self.roads)
                self.spots = original_spots
                self.roads = original_roads
            
        return answer_spots, answer_roads
    
    def resource_update(self,p1,p2):
        """
        Update inventory of players with resources based on random throw and current board configuration 
        """       
        self.throw = random.randint(1, 6) + random.randint(1, 6)
        for i in range(len(self.spots)):
            if self.spots[i] == self.P1_S:
                for j in range(len(self.resource[i])):
                    if self.value[i][j] == self.throw:
                        if self.resource[i][j] == "Lumber":
                            p1.lumber += 1
                        elif self.resource[i][j] == "Brick":
                            p1.brick += 1
                        elif self.resource[i][j] == "Wool":
                            p1.wool += 1
                        elif self.resource[i][j] == "Grain":
                            p1.grain += 1
                        elif self.resource[i][j] == "Ore":
                            p1.ore += 1

            elif self.spots[i] == self.P2_S:
                for j in range(len(self.resource[i])):
                    if self.value[i][j] == self.throw:
                        if self.resource[i][j] == "Lumber":
                            p2.lumber += 1
                        elif self.resource[i][j] == "Brick":
                            p2.brick += 1
                        elif self.resource[i][j] == "Wool":
                            p2.wool += 1
                        elif self.resource[i][j] == "Grain":
                            p2.grain += 1
                        elif self.resource[i][j] == "Ore":
                            p2.ore += 1                
            
            elif self.spots[i] == self.P1_C:
                for j in range(len(self.resource[i])):
                    if self.value[i][j] == self.throw:
                        if self.resource[i][j] == "Lumber":
                            p1.lumber += 2
                        elif self.resource[i][j] == "Brick":
                            p1.brick += 2
                        elif self.resource[i][j] == "Wool":
                            p1.wool += 2
                        elif self.resource[i][j] == "Grain":
                            p1.grain += 2
                        elif self.resource[i][j] == "Ore":
                            p1.ore += 2 
            
            elif self.spots[i] == self.P2_C:
                for j in range(len(self.resource[i])):
                    if self.value[i][j] == self.throw:
                        if self.resource[i][j] == "Lumber":
                            p2.lumber += 2
                        elif self.resource[i][j] == "Brick":
                            p2.brick += 2
                        elif self.resource[i][j] == "Wool":
                            p2.wool += 2
                        elif self.resource[i][j] == "Grain":
                            p2.grain += 2
                        elif self.resource[i][j] == "Ore":
                            p2.ore += 2 
  
        return self

    def map_update(self,rounds,round,num_games):
        """
        Generate map based on current board configuration
        """       
        plt.figure()
        img = plt.imread("board.jpg")
        plt.imshow(img, zorder=0, extent=[-5, 5, -8, 8])
        for i in range(len(self.spots)):
            if self.spots[i] == self.EMPTY_SPOT:
                plt.scatter(self.xloc[i],self.yloc[i], color="black", marker="s",label=("Available Spots"))
            if self.spots[i] == self.P1_S:
                plt.scatter(self.xloc[i],self.yloc[i], color="green", marker="s",label=("Settlement Player 1"))
            if self.spots[i] == self.P1_C:
                plt.scatter(self.xloc[i],self.yloc[i], color="green", marker="D",label=("City Player 1"))
            if self.spots[i] == self.P2_S:
                plt.scatter(self.xloc[i],self.yloc[i], color="red", marker="s",label=("Settlement Player 2"))
            if self.spots[i] == self.P2_C:
                plt.scatter(self.xloc[i],self.yloc[i], color="red", marker="D",label=("City Player 2"))
            for j in range(len(self.roads[i])):
                if self.roads[i][j] == self.P1_S:
                    plt.scatter((self.xloc[i]*2+self.xloc[self.neighbours[i][j]])/3,(self.yloc[i]*2+self.yloc[self.neighbours[i][j]])/3, color="green", marker="_",label=("Road Player 1"))
                if self.roads[i][j] == self.P2_S:
                    plt.scatter((self.xloc[i]*2+self.xloc[self.neighbours[i][j]])/3,(self.yloc[i]*2+self.yloc[self.neighbours[i][j]])/3, color="red", marker="_",label=("Road Player 2"))

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),loc=0,bbox_to_anchor =(1.0, 1.0))    
        plt.grid(False)
        plt.savefig(os.path.join(save_folder,f"Board-{rounds}-{round}-{num_games}.png"))
        plt.close()
  