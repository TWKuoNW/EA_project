import gymnasium as gym
import gym_pusht
import numpy as np
import pygame
from datetime import datetime
from PIL import Image   # Pillow 用來存 PNG

# --------------------------------------------------
# 建立 PushT 環境：obs_type="state" 才能拿到 agent 座標
# render_mode="rgb_array" 讓我們自己畫畫面
# --------------------------------------------------
env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")

# 固定初始狀態：agent_x, agent_y, block_x, block_y, block_angle
fixed_state = [20.0, 250.0, 100.0, 200.0, 0.0]
obs, info = env.reset(options={"reset_to_state": fixed_state})

# obs = [agent_x, agent_y, block_x, block_y, block_angle]
current_target = np.array(obs[:2], dtype=np.float32)  # 目前要移動到的 [x, y]

# 先 render 一張圖，決定視窗大小
frame = env.render()  # shape: (H, W, 3), dtype=uint8
H, W, _ = frame.shape

# --------------------------------------------------
# 初始化 pygame 視窗
# --------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("PushT Keyboard Control (Arrow keys, S = screenshot, ESC = quit)")

clock = pygame.time.Clock()

# 取得 action space 範圍，理論上是 [0, 512]
action_low = env.action_space.low.astype(np.float32)
action_high = env.action_space.high.astype(np.float32)

# 每次按鍵移動多少畫素（可以自行調大調小）
STEP = 20.0

print("操作說明：")
print("  ↑ ↓ ← →：控制 end-effector 移動")
print("  S       ：截圖（screenshot_YYYYMMDD_HHMMSS.png）")
print("  ESC / 關視窗：離開")

running = True
s_pressed_last = False  # 記錄上一幀 S 是否有被按住

while running:
    # --------------------------------------------------
    # 顯示目前畫面
    # --------------------------------------------------
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))  # (W, H, 3)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    action_changed = False  # 這一輪有沒有按到方向鍵

    # --------------------------------------------------
    # 處理視窗事件（關閉、ESC、方向鍵）
    # --------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            # 離開
            if event.key == pygame.K_ESCAPE:
                running = False

            # 方向鍵：更新 current_target（絕對座標）
            elif event.key == pygame.K_UP:
                # 往上：y 變小
                current_target[1] -= STEP
                action_changed = True

            elif event.key == pygame.K_DOWN:
                # 往下：y 變大
                current_target[1] += STEP
                action_changed = True

            elif event.key == pygame.K_LEFT:
                # 往左：x 變小
                current_target[0] -= STEP
                action_changed = True

            elif event.key == pygame.K_RIGHT:
                # 往右：x 變大
                current_target[0] += STEP
                action_changed = True

    # --------------------------------------------------
    # 用 get_pressed 輪詢 S 是否被按下 → 截圖
    # （避免 KEYDOWN 沒被吃到的問題）
    # --------------------------------------------------
    keys = pygame.key.get_pressed()
    if keys[pygame.K_s]:
        if not s_pressed_last:
            # 這一幀是「新按下 S」：存一張圖
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            img = Image.fromarray(frame.astype(np.uint8))
            img.save(filename)
            print(f"Saved screenshot: {filename}")
        s_pressed_last = True
    else:
        s_pressed_last = False

    # --------------------------------------------------
    # 如果這一輪有按方向鍵，就 step 一次環境
    # --------------------------------------------------
    if action_changed:
        # 把 target 限制在合法範圍內 [0, 512]
        current_target = np.clip(current_target, action_low, action_high)

        action = current_target.astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # 更新畫面
        frame = env.render()

        # 同步 current_target = 實際新的 agent 位置
        current_target = np.array(obs[:2], dtype=np.float32)

        # episode 結束就 reset 回固定狀態
        if terminated or truncated:
            obs, info = env.reset(options={"reset_to_state": fixed_state})
            frame = env.render()
            current_target = np.array(obs[:2], dtype=np.float32)

    # 控制更新頻率（60 FPS 左右）
    clock.tick(60)

env.close()
pygame.quit()
