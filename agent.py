import torch
import random
import numpy as np
from collections import deque #data structure
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot,write

MAX_MEMORY= 100_000
BATCH_SIZE = 1000

LR = 0.001


#randomness=False #False if i want to use preload model
randomness=True #True if i want to start from scratch


class Agent:
    def __init__(self):
        self.n_games= 0
        self.epsilon = 0 #randomness parameter
        self.gamma = 0.9  #discount rate default is 0.9
        self.memory = deque(maxlen=MAX_MEMORY) #automatically remove left element like popleft()

        #inizializzo il modello 11 input, 3 output
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)

    def get_state(self,game):
        #11 values 
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) #per convertire quei true e false in 0,1

    def remember(self,state,action,reward,next_state,done):
         self.memory.append((state,action,reward,next_state,done)) # popleft if MAXMEM IS REACHED

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            #we estract 1000 samples from the memory 
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory

        states,actions,rewards,next_states,dones = zip (*mini_sample)

        self.trainer.train_step(states,actions,rewards,next_states,dones)


    def train_short_memory(self,state,action,reward,next_state,done ):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # random moves tradeoff between exploration and exploitation
        # se volessi sfruttare il modello gia allenato dovrei rimuovere 
        # la randomness con la epsilon
        self.epsilon = 80 - self.n_games # piu games meno random
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0=torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
    #abbiamo 3 valori vogliamo il max
            move = torch.argmax(prediction).item()
            #print(prediction)
            final_move[move] = 1
        return final_move

def train():
    plot_scores=[]
    plot_mean_scores = []
    total_score = 0
    record = 0 

    agent = Agent()
    agent.model.load()

    game = SnakeGameAI()


    while True:
        #get old state
        state_old=agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new=agent.get_state(game)
        
        #train short memory
        agent.train_short_memory(state_old,final_move,reward, state_new,done)
        
        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            #train long memory ??? experience replay 
            game.reset()
            agent.n_games +=1
            print("Epsilon random factor",80 - agent.n_games)
            agent.train_long_memory()
            
            if score > record:
                record = score 
                #salva il model soltanto se l high score é maggiore del record 
                agent.model.save()

            print('Game ',agent.n_games,'Score ',score,'Record ',record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            #plot_mean_scores.append(mean_score)
            #plot(plot_scores,plot_mean_scores)
            
            write(agent.n_games,score,mean_score)

if __name__ == '__main__':
    train()
