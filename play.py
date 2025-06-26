# play.py (已修改为15x15版本)
import pygame
import sys
import os
import time
import numpy as np

# 假设 train.py 在同一目录下或在Python路径中
# 如果不在，您可能需要调整导入方式
from train import Board, MCTSPlayer, PolicyValueNet

# --- 核心常量 ---
BOARD_WIDTH = 6    ### <<< 已修改为 15
BOARD_HEIGHT = 6   ### <<< 已修改为 15
N_IN_ROW = 5
MODEL_FOLDER = 'alpha_gomoku_pytorch_models'
# 注意: 请确保加载的模型是为15x15棋盘训练的，否则AI性能会很差
MODEL_PATH = r"C:\Users\ycf15\Desktop\Gomoku\test\best_policy91400g200.model" # 建议加载最优模型

# --- 美化GUI设置 ---
GRID_SIZE = 40        ### <<< 已修改，以适应15x15棋盘
BORDER = 50
WINDOW_WIDTH = BOARD_WIDTH * GRID_SIZE + 2 * BORDER
WINDOW_HEIGHT = BOARD_HEIGHT * GRID_SIZE + 2 * BORDER
PIECE_RADIUS = GRID_SIZE // 2 - 3 # 调整棋子大小以适应新格子

# --- 颜色主题 ---
BOARD_COLOR_LIGHT = (222, 184, 135)
BOARD_COLOR_DARK = (205, 170, 125)
LINE_COLOR = (80, 80, 80)
BLACK_COLOR = (10, 10, 10)
WHITE_COLOR = (245, 245, 245)
SHADOW_COLOR = (50, 50, 50, 100)
TEXT_COLOR = (40, 40, 40)
HOVER_COLOR_BLACK = (0, 0, 0, 120)
HOVER_COLOR_WHITE = (255, 255, 255, 120)

class GameGUI:
    def __init__(self, board):
        self.board = board
        self.human_player_id = None
        self.ai_player = None
        pygame.init()
        pygame.display.set_caption("Gomoku Pro (15x15) - 人机对战") ### <<< 已修改
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.load_font()

    def load_font(self):
        font_paths = ['SourceHanSans.otf', 'msyh.ttf', 'simhei.ttf']
        font_found = False
        for path in font_paths:
            if os.path.exists(path):
                self.font = pygame.font.Font(path, 30)
                font_found = True
                break
        if not font_found:
            print("警告: 未找到中文字体。请下载'Source Han Sans'并命名为'SourceHanSans.otf'放在脚本同目录下。")
            self.font = pygame.font.Font(None, 40)

    def draw_board(self):
        # 绘制有从上到下渐变效果的棋盘背景
        for y in range(WINDOW_HEIGHT):
            r = np.interp(y, [0, WINDOW_HEIGHT], [BOARD_COLOR_LIGHT[0], BOARD_COLOR_DARK[0]])
            g = np.interp(y, [0, WINDOW_HEIGHT], [BOARD_COLOR_LIGHT[1], BOARD_COLOR_DARK[1]])
            b = np.interp(y, [0, WINDOW_HEIGHT], [BOARD_COLOR_LIGHT[2], BOARD_COLOR_DARK[2]])
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))

        # 绘制棋盘网格线
        for i in range(BOARD_WIDTH):
            pygame.draw.line(self.screen, LINE_COLOR, (BORDER + i * GRID_SIZE, BORDER), (BORDER + i * GRID_SIZE, WINDOW_HEIGHT - BORDER))
        for i in range(BOARD_HEIGHT):
            pygame.draw.line(self.screen, LINE_COLOR, (BORDER, BORDER + i * GRID_SIZE), (WINDOW_WIDTH - BORDER, BORDER + i * GRID_SIZE))

        # 绘制15x15棋盘的标准星位
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)] ### <<< 已修改
        for r, c in star_points:
            if r < self.board.height and c < self.board.width:
                center = (BORDER + c * GRID_SIZE, BORDER + r * GRID_SIZE)
                pygame.draw.circle(self.screen, LINE_COLOR, center, 6)

    def draw_pieces(self):
        for move, player_id in self.board.states.items():
            row, col = move // self.board.width, move % self.board.width
            center = (BORDER + col * GRID_SIZE, BORDER + row * GRID_SIZE)
            
            # 绘制阴影，增加立体感
            shadow_center = (center[0] + 3, center[1] + 3)
            s_shadow = pygame.Surface((PIECE_RADIUS*2, PIECE_RADIUS*2), pygame.SRCALPHA)
            pygame.draw.circle(s_shadow, SHADOW_COLOR, (PIECE_RADIUS, PIECE_RADIUS), PIECE_RADIUS)
            self.screen.blit(s_shadow, (shadow_center[0] - PIECE_RADIUS, shadow_center[1] - PIECE_RADIUS))
            
            # 绘制棋子
            color = BLACK_COLOR if player_id == 1 else WHITE_COLOR
            pygame.draw.circle(self.screen, color, center, PIECE_RADIUS)

    def draw_hover_preview(self):
        # 只在轮到人类玩家时显示悬停预览
        if not self.game_over and self.board.get_current_player() == self.human_player_id:
            try:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                # 精确计算鼠标所在的行列
                col = int((mouse_x - BORDER + GRID_SIZE / 2) // GRID_SIZE)
                row = int((mouse_y - BORDER + GRID_SIZE / 2) // GRID_SIZE)
                if 0 <= col < self.board.width and 0 <= row < self.board.height:
                    move = self.board.width * row + col
                    if move in self.board.availables:
                        color = HOVER_COLOR_BLACK if self.human_player_id == 1 else HOVER_COLOR_WHITE
                        center = (BORDER + col * GRID_SIZE, BORDER + row * GRID_SIZE)
                        s = pygame.Surface((PIECE_RADIUS*2, PIECE_RADIUS*2), pygame.SRCALPHA)
                        pygame.draw.circle(s, color, (PIECE_RADIUS, PIECE_RADIUS), PIECE_RADIUS)
                        self.screen.blit(s, (center[0] - PIECE_RADIUS, center[1] - PIECE_RADIUS))
            except pygame.error: pass # 忽略鼠标在窗口外的错误

    def show_message(self, text, size=30, top_offset=25, color=TEXT_COLOR):
        # 在窗口顶部中央显示消息
        try: font = pygame.font.Font(self.font.get_path(), size)
        except AttributeError: font = self.font
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH / 2, top_offset))
        self.screen.blit(text_surface, text_rect)

    def run(self):
        try:
            # 加载为15x15棋盘训练的AI模型
            policy_value_net = PolicyValueNet(BOARD_WIDTH, BOARD_HEIGHT, model_file=MODEL_PATH)
            # is_selfplay=0 表示这是在对战，而不是自我训练
            self.ai_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=800, is_selfplay=0)
        except Exception as e:
            print(f"错误: 无法加载AI模型: {e}")
            print(f"请确认 '{MODEL_PATH}' 是否存在并且是为15x15棋盘训练的。")
            sys.exit(1)

        # 游戏开始前，让玩家选择颜色
        choice = ''
        while choice not in ['1', '2']:
            choice = input("请选择您的颜色，输入 '1' 为执黑先手, 输入 '2' for a 15x15 board 为执白后手: ")
        self.human_player_id = int(choice)
        self.ai_player.set_player_ind(2 if self.human_player_id == 1 else 1)
        
        self.board.init_board()
        self.game_over = False

        while True:
            # --- 事件处理循环 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # 人类玩家落子
                if not self.game_over and self.board.get_current_player() == self.human_player_id:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mouse_x, mouse_y = event.pos
                        col = int((mouse_x - BORDER + GRID_SIZE / 2) // GRID_SIZE)
                        row = int((mouse_y - BORDER + GRID_SIZE / 2) // GRID_SIZE)
                        if 0 <= col < self.board.width and 0 <= row < self.board.height:
                            move = self.board.width * row + col
                            if move in self.board.availables:
                                self.board.do_move(move)
                                self.game_over, _ = self.board.game_end()
            
            # --- AI玩家落子 ---
            if not self.game_over and self.board.get_current_player() == self.ai_player.player:
                self.draw_board(); self.draw_pieces(); self.show_message("AI 正在思考...")
                pygame.display.flip() # 立即更新屏幕显示"AI思考中"
                
                # temp=0 表示AI会选择最稳健（即MCTS访问次数最多）的走法
                ai_move = self.ai_player.get_action(self.board, temp=0, return_prob=0)
                if ai_move is not None:
                    self.board.do_move(ai_move)
                    self.game_over, _ = self.board.game_end()
            
            # --- 绘制屏幕 ---
            self.draw_board()
            self.draw_pieces()
            self.draw_hover_preview()

            # --- 显示游戏状态信息 ---
            if self.game_over:
                _, winner = self.board.game_end()
                if winner == self.human_player_id:
                    self.show_message("恭喜, 您赢了!")
                elif winner == self.ai_player.player:
                    self.show_message("游戏结束: AI获胜!")
                else:
                    self.show_message("游戏结束: 平局!")
            else:
                if self.board.get_current_player() == self.human_player_id:
                    self.show_message("您的回合")

            pygame.display.flip()
            time.sleep(0.05) # 降低CPU占用

if __name__ == '__main__':
    board = Board(width=BOARD_WIDTH, height=BOARD_HEIGHT, n_in_row=N_IN_ROW)
    game_gui = GameGUI(board)
    game_gui.run()