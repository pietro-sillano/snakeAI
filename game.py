import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np



pygame.init() # per inizilizzare i moduli correttamente
font = pygame.font.Font('arial.ttf',25) #faster implement

# cose da cambiare da clean a AI
# reset every game
# reward for our agent
# play(action) --> direction instead of the key event
# game_iteration
# is_collision 


class Direction(Enum):
  #é un modo per evitare errori per la direzione in questo modo dovrebbe solo 
  #considerare questi 4 valori
    RIGHT=1
    LEFT =2
    UP = 3
    DOWN =4

Point = namedtuple('Point','x,y')  # self.head = [self.w,self.h] questo potrebbe essere fonte di errore quindi usamo un altro metodo

# Const
BLOCK_SIZE = 20 #ogni blocchetto di snalke sará 20 pixels
SPEED = 40
# RGB colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)

class SnakeGameAI:

  def __init__(self,w=640,h=480):
    #init display
    self.w=w
    self.h=h
    self.display=pygame.display.set_mode((self.w,self.h))
    pygame.display.set_caption('Snake')
    self.clock=pygame.time.Clock() #per la velocitá del nostro gioco
     
    self.reset()

  def reset(self):
    #init game state
    self.direction = Direction.RIGHT
    self.head = Point(self.w/2,self.h/2) #mettiamo la testa al centro del display
    #per il corpo usiamo una lista: testa, primo quadratino, secondo quadratino
    self.snake = [self.head, 
                  Point(self.head.x-BLOCK_SIZE,self.head.y),
                  Point(self.head.x-(2*BLOCK_SIZE),self.head.y)] 

    self.score = 0 
    self.food= None
    #questa qua sotto é un helper function che non ho idea di cosa sia :(
    self._place_food() 
    self.frame_iteration=0
     
  def _place_food(self):
    #lo scrivo cosi per piazzare il cibo in una posizione multipla intera del blocksize
    x=random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
    y=random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
    self.food = Point(x,y)

    #vogliamo controllare che il cibo non sia piazzato dentro il serpente quindi check
    #lo faccio ricorsivamente in modo che generi cibo finche non é fuori dal serpente

    if self.food in self.snake:
      self._place_food()

  def play_step(self,action):
    self.frame_iteration += 1
    # 1.collect user input
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
          pygame.quit()
          quit()
     
    # 2. move
    self._move(action) #update the head
    self.snake.insert(0,self.head)

    # 3. check if over
    game_over = False
    reward =0
    if self.is_collision() or (self.frame_iteration > 100*len(self.snake)):  #default is 100
      game_over = True
      reward = -10 
      return reward, game_over, self.score
    

    # 4. place new food or just move
    if self.head == self.food:
      self.score+=1
      self._place_food()
      reward = 10
    else:
      self.snake.pop() #rimuove l ultimo quadratino del snake

    # 5. update ui and clock
    self._update_ui()
    self.clock.tick(SPEED)

    # 6. return game over and score
    return reward, game_over, self.score  

  def _move(self,action ):
    # [straight,right,left]

    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT,Direction.UP]
    idx = clock_wise.index(self.direction)

    if np.array_equal(action, [1,0,0]):
      new_dir=clock_wise[idx] # no change

    elif np.array_equal(action, [0,1,0]):
      next_idx = (idx+1) % 4
      new_dir=clock_wise[next_idx] # right turn r -> d -> l -> u

    else: # [0,0,1]
      next_idx = (idx-1) % 4
      new_dir=clock_wise[next_idx] # right turn r -> d -> l -> u

    self.direction = new_dir

    x=self.head.x
    y=self.head.y

    if self.direction == Direction.RIGHT:
      x += BLOCK_SIZE
    elif self.direction == Direction.LEFT:
      x -= BLOCK_SIZE
    elif self.direction == Direction.UP:
      y -= BLOCK_SIZE
    elif self.direction == Direction.DOWN :
      y += BLOCK_SIZE

    self.head=Point(x,y)

  def is_collision(self, pt=None):
    if pt is None:
      pt = self.head
    #hits the boundary 
    if pt.x >self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
      return True
    #hits itself 
    if self.head in self.snake[1:]: #we want to check if the head is contain in the snake positions except for the head
      return True

    return False

  def _update_ui(self):
    self.display.fill(BLACK )

    for pt in self.snake:
      pygame.draw.rect(self.display,BLUE1,pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
      pygame.draw.rect(self.display,BLUE2,pygame.Rect(pt.x+4,pt.y+4,12,12))
    
    pygame.draw.rect(self.display,RED,pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))

    text = font.render("Score: "+ str(self.score),True, WHITE)
    self.display.blit(text, [0,0])
    
    moves = font.render("Moves : "+ str(self.frame_iteration),True, WHITE)
    self.display.blit(moves, [470,0])

    moves_left = font.render("Moves left : "+ str((100*len(self.snake)+1)-self.frame_iteration),True, WHITE)
    self.display.blit(moves_left, [450,40])

    pygame.display.flip()