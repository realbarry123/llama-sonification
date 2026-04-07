import pygame
import sounddevice as sd
from sonifier import Sonifier
from model_wrapper import ModelWrapper
import threading
import queue

FULL_SCREEN = False

# MODEL SETUP

sonifier = Sonifier((1, 17, 2048), 2/17, fs=44100)
model = ModelWrapper()
model.seed("Hello, world")


# PRODUCER
# producer-consumer pattern (very nice)

audio_queue = queue.Queue(maxsize=2)

def producer():
    while True:
        token, states = model.next()
        audio = sonifier(states)
        audio_queue.put((token, audio))  # blocks if queue is full

producer_thread = threading.Thread(target=producer, daemon=True)
producer_thread.start()


# GRAPHICS SETUP

pygame.init()
VW = 500
VH = 500

if FULL_SCREEN:
    VW = 1512
    VH = 982
INIT_TEXT_COLOR = 255
INIT_D_TEXT_COLOR = -5
DD_TEXT_COLOR = 0.05

screen = pygame.display.set_mode((VW, VH))
clock = pygame.time.Clock()
if FULL_SCREEN: 
    pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
running = True
font = pygame.font.SysFont('Arial', 128)
debug_font = pygame.font.SysFont('Arial', 15)
current_token = ""
text_color = INIT_TEXT_COLOR
d_text_color = INIT_D_TEXT_COLOR

while running:

    # EVENTS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            model.write_history("context.txt")

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LSHIFT] and keys[pygame.K_9] and keys[pygame.K_SEMICOLON]:
        running = False
        model.write_history("context.txt")
    elif keys[pygame.K_ESCAPE]:
        model.write_history("context.txt")
        model.__init__()
        model.seed("")

    # CONSUMER (very nice)

    is_playing = False  # placeholder

    try:
        is_playing = sd.get_stream().active

    except RuntimeError:
        is_playing = False

    if not is_playing and not audio_queue.empty():
        current_token, audio = audio_queue.get()
        sd.play(audio)
        text_color = INIT_TEXT_COLOR
        d_text_color = INIT_D_TEXT_COLOR

    screen.fill("black")

    text_surface = font.render(current_token, True, (text_color, text_color, text_color))
    text_rect = text_surface.get_rect()
    text_rect.center = (VW // 2, VH // 2)
    screen.blit(text_surface, text_rect)

    # debug_surface = debug_font.render(str(pygame.time.get_ticks()), True, (255, 0, 255))
    # screen.blit(debug_surface, text_rect)

    pygame.display.flip()

    d_text_color = min(d_text_color + DD_TEXT_COLOR, 0)
    text_color = max(text_color + d_text_color, 0)

    clock.tick(60)  # limits FPS to 60

pygame.quit()