import pygame
import random
import time


pygame.init()


screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("EEG Hand Thinking Game")


left_hand_img = pygame.image.load("C:\\Users\\ivslo\\Desktop\\left_hand.png")
right_hand_img = pygame.image.load("C:\\Users\\ivslo\\Desktop\\right_hand.jpg")
neutral_img = pygame.image.load("C:\\Users\\ivslo\\Desktop\\white_background.jpg")


left_hand_img = pygame.transform.scale(left_hand_img, (400, 400))
right_hand_img = pygame.transform.scale(right_hand_img, (400, 400))
neutral_img = pygame.transform.scale(neutral_img, (400, 400))


font = pygame.font.Font(None, 36)


score = 0
total_rounds = 0
display_time = 5
feedback_display_time = 5
current_state = None


def get_eeg_input():
    return random.choice([0, 1, 2])


def title_screen():
    screen.fill((255, 255, 255))
    title_text = font.render("EEG Hand Thinking Game", True, (0, 0, 0))
    instructions_text = font.render("Focus on the displayed hand or relax", True, (0, 0, 0))
    start_text = font.render("Press Space to Start", True, (0, 0, 0))
    screen.blit(title_text, (screen_width // 2 - title_text.get_width() // 2, 100))
    screen.blit(instructions_text, (screen_width // 2 - instructions_text.get_width() // 2, 200))
    screen.blit(start_text, (screen_width // 2 - start_text.get_width() // 2, 300))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False


running = True
while running:
    title_screen()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        current_state = random.choice([0, 1, 2])


        screen.fill((255, 255, 255))
        if current_state == 0:
            screen.blit(left_hand_img, (200, 100))
        elif current_state == 1:
            screen.blit(right_hand_img, (200, 100))
        else:
            screen.blit(neutral_img, (200, 100))
            neutral_text = font.render("Neutral State: Relax and think of nothing", True, (0, 0, 255))
            screen.blit(neutral_text, (screen_width // 2 - neutral_text.get_width() // 2, screen_height // 2 + 200))


        if total_rounds % 5 == 0:
            score_text = font.render(f"Score: {score}", True, (0, 0, 0))
            round_text = font.render(f"Round: {total_rounds}", True, (0, 0, 0))
            screen.blit(score_text, (20, 20))
            screen.blit(round_text, (20, 50))

        pygame.display.flip()


        time.sleep(display_time)


        eeg_input = get_eeg_input()


        if eeg_input == current_state:
            result_text = "Correct! Thinking matches displayed state."
            score += 1
        else:
            result_text = "Incorrect. Try to focus on the displayed state."

        total_rounds += 1
        accuracy = (score / total_rounds) * 100


        screen.fill((255, 255, 255))
        feedback_surface = font.render(result_text, True, (0, 128, 0))
        accuracy_surface = font.render(f"Accuracy: {accuracy:.2f}%", True, (0, 128, 0))
        screen.blit(feedback_surface, (screen_width // 2 - feedback_surface.get_width() // 2, screen_height // 2 - 50))
        screen.blit(accuracy_surface, (screen_width // 2 - accuracy_surface.get_width() // 2, screen_height // 2 + 50))
        pygame.display.flip()

        time.sleep(feedback_display_time)


pygame.quit()