# Example file showing a basic pygame "game loop"
import pygame
import sounddevice as sd
from sonifier import Sonifier
from model_wrapper import ModelWrapper

# VARIABLES

sonifier = Sonifier((1, 17, 2048), 1/17, fs=44100)
model = ModelWrapper()
model.seed("I")

# GRAPHICS SETUP

pygame.init()
VW = 3024//2
VH = 1964//2
screen = pygame.display.set_mode((VW, VH))
clock = pygame.time.Clock()
pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
running = True
font = pygame.font.SysFont('Arial', 48)
debug_font = pygame.font.SysFont('Arial', 15)
last_ticked_time = 0

while running:

    # EVENTS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LSHIFT] and keys[pygame.K_9] and keys[pygame.K_SEMICOLON]:
        running = False
    
    if pygame.time.get_ticks() - last_ticked_time >= 1000: # tick every second

        # COMPUTATIONS

        token, states = model.next()
        sd.play(sonifier(states))


        # GRAPHICS

        screen.fill("black")

        text_surface = font.render(token, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.center = (VW // 2, VH // 2)
        screen.blit(text_surface, text_rect)

        debug_surface = debug_font.render(str(pygame.time.get_ticks()), True, (255, 0, 255))
        last_ticked_time = pygame.time.get_ticks()
        screen.blit(debug_surface, text_rect)
    
        pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()