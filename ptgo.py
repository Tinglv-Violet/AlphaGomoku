import random
import copy
import os
from collections import defaultdict, deque
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame
from pygame.locals import *


class Board(object):
    """棋盘游戏逻辑控制"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get("width", 15))  # 棋盘宽度
        self.height = int(kwargs.get("height", 15))  # 棋盘高度
        self.states = {}  # 棋盘状态为一个字典,键: 移动步数,值: 玩家的棋子类型
        self.n_in_row = int(kwargs.get("n_in_row", 5))  # 5个棋子一条线则获胜
        self.players = [1, 2]  # 玩家1,2

    def init_board(self, start_player=0):
        # 初始化棋盘

        # 当前棋盘的宽高小于5时,抛出异常(因为是五子棋)
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception("棋盘的长宽不能少于{}".format(self.n_in_row))
        self.current_player = self.players[start_player]  # 先手玩家
        self.availables = list(
            range(self.width * self.height)
        )  # 初始化可用的位置列表
        self.states = {}  # 初始化棋盘状态
        self.last_move = -1  # 初始化最后一次的移动位置

    def current_state(self):
        """
        从当前玩家的角度返回棋盘状态。
        状态形式: 4 * 宽 * 高
        """
        # 使用4个15x15的二值特征平面来描述当前的局面
        # 前两个平面分别表示当前player的棋子位置和对手player的棋子位置，有棋子的位置是1，没棋子的位置是0
        # 第三个平面表示对手player最近一步的落子位置，也就是整个平面只有一个位置是1，其余全部是0
        # 第四个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[
                players == self.current_player
            ]  # 获取棋盘状态上属于当前玩家的所有移动值
            move_oppo = moves[
                players != self.current_player
            ]  # 获取棋盘状态上属于对方玩家的所有移动值
            square_state[0][
                move_curr // self.width,  # 对第一个特征平面填充值(当前玩家)
                move_curr % self.height,
            ] = 1.0
            square_state[1][
                move_oppo // self.width,  # 对第二个特征平面填充值(对方玩家)
                move_oppo % self.height,
            ] = 1.0
            # 指出最后一个移动位置
            square_state[2][
                self.last_move
                // self.width,  # 对第三个特征平面填充值(对手最近一次的落子位置)
                self.last_move % self.height,
            ] = 1.0
        if (
            len(self.states) % 2 == 0
        ):  # 对第四个特征平面填充值,当前玩家是先手,则填充全1,否则为全0
            square_state[3][:, :] = 1.0
        # 将每个平面棋盘状态按行逆序转换(第一行换到最后一行,第二行换到倒数第二行..)
        return square_state[:, ::-1, :]

    def do_move(self, move):
        # 根据移动的数据更新各参数
        self.states[move] = self.current_player  # 将当前的参数存入棋盘状态中
        self.availables.remove(move)  # 从可用的棋盘列表移除当前移动的位置
        self.current_player = (
            self.players[0]
            if self.current_player == self.players[1]
            else self.players[1]
        )  # 改变当前玩家
        self.last_move = move  # 记录最后一次的移动位置

    def has_a_winner(self):
        # 是否产生赢家
        width = self.width  # 棋盘宽度
        height = self.height  # 棋盘高度
        states = self.states  # 状态
        n = self.n_in_row  # 获胜需要的棋子数量

        # 当前棋盘上所有的落子位置
        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            # 当前棋盘落子数在7个以上时会产生赢家,落子数低于7个时,直接返回没有赢家
            return False, -1

        # 遍历落子数
        for m in moved:
            h = m // width
            w = m % width  # 获得棋子的坐标
            player = states[m]  # 根据移动的点确认玩家

            # 判断各种赢棋的情况
            # 横向5个
            if (
                w in range(width - n + 1)
                and len(set(states.get(i, -1) for i in range(m, m + n))) == 1
            ):
                return True, player

            # 纵向5个
            if (
                h in range(height - n + 1)
                and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * width, width)
                    )
                )
                == 1
            ):
                return True, player

            # 左上到右下斜向5个
            if (
                w in range(width - n + 1)
                and h in range(height - n + 1)
                and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * (width + 1), width + 1)
                    )
                )
                == 1
            ):
                return True, player

            # 右上到左下斜向5个
            if (
                w in range(n - 1, width)
                and h in range(height - n + 1)
                and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * (width - 1), width - 1)
                    )
                )
                == 1
            ):
                return True, player

        # 当前都没有赢家,返回False
        return False, -1

    def game_end(self):
        """检查当前棋局是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            # 棋局布满,没有赢家
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


# 加上UI的布局的训练方式
class Game_UI(object):
    """游戏控制区域"""

    def __init__(self, board, **kwargs):
        self.board = board  # 加载棋盘控制类

        # 初始化 pygame
        pygame.init()

    def start_play_evaluate(self, player1, player2, start_player=0):
        """开始一局游戏，评估当前的价值策略网络的胜率"""
        if start_player not in (0, 1):
            # 如果玩家不在玩家1,玩家2之间,抛出异常
            raise Exception("开始的玩家必须为0(玩家1)或1(玩家2)")
        self.board.init_board(start_player)  # 初始化棋盘
        p1, p2 = self.board.players  # 加载玩家1,玩家2
        player1.set_player_ind(p1)  # 设置玩家1
        player2.set_player_ind(p2)  # 设置玩家2
        players = {p1: player1, p2: player2}

        while True:
            current_player = self.board.current_player  # 获取当前玩家
            player_in_turn = players[current_player]  # 当前玩家的信息
            move = player_in_turn.get_action(
                self.board
            )  # 基于MCTS的AI下一步落子
            self.board.do_move(move)  # 根据下一步落子的状态更新棋盘各参数

            # 判断当前棋局是否结束
            end, winner = self.board.game_end()
            # 结束
            if end:
                win = winner
                break

        return win

    def start_play_train(self, player, temp=1e-3):
        """
        开始自我博弈，使用MCTS玩家开始自己玩游戏,重新使用搜索树并存储自己玩游戏的数据
        (state, mcts_probs, z) 提供训练
        """
        self.board.init_board()  # 初始化棋盘
        states, mcts_probs, current_players = (
            [],
            [],
            [],
        )  # 状态,mcts的行为概率,当前玩家

        while True:
            # 根据当前棋盘状态返回可能得行为,及行为对应的概率
            move, move_probs = player.get_action(
                self.board, temp=temp, return_prob=1
            )
            # 存储数据
            states.append(self.board.current_state())  # 存储状态数据
            mcts_probs.append(move_probs)  # 存储行为概率数据
            current_players.append(self.board.current_player)  # 存储当前玩家
            # 执行一个移动
            self.board.do_move(move)

            # 判断该局游戏是否终止
            end, winner = self.board.game_end()
            if end:
                # 从每个状态的当时的玩家的角度看待赢家
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    # 没有赢家时
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MSCT的根节点
                player.reset_player()
                return winner, zip(states, mcts_probs, winners_z)


class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # 公共网络层
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        # 行动策略网络层
        self.act_conv1 = nn.Conv2d(
            in_channels=128, out_channels=4, kernel_size=1, padding=0
        )
        self.act_fc1 = nn.Linear(
            4 * self.board_width * self.board_height,
            self.board_width * self.board_height,
        )
        # 状态价值网络层
        self.val_conv1 = nn.Conv2d(
            in_channels=128, out_channels=2, kernel_size=1, padding=0
        )
        self.val_fc1 = nn.Linear(2 * self.board_width * self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, inputs):
        # 公共网络层
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 行动策略网络层
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_height * self.board_width)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # 状态价值网络层
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_height * self.board_width)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet:
    """策略&值网络"""

    def __init__(
        self, board_width, board_height, model_file=None, use_gpu=True
    ):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-3  # coef of l2 penalty

        # 设置device
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.policy_value_net = Net(self.board_width, self.board_height).to(
            self.device
        )

        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            lr=0.02,
            weight_decay=self.l2_const,
        )

        if model_file:
            try:
                net_params = torch.load(model_file, map_location=self.device)
                self.policy_value_net.load_state_dict(net_params)
            except Exception as e:
                print(f"Could not load model: {e}")


    def policy_value_evaluate(self, state_batch):
        """
        评估函数
        Args:
            input: 一组棋盘状态
            output: 根据棋盘状态输出对应的动作概率及价值
        """
        self.policy_value_net.eval()
        state_batch_tensor = torch.from_numpy(state_batch).to(self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch_tensor)
        act_probs = np.exp(log_act_probs.cpu().numpy())
        return act_probs, value.cpu().numpy()

    def policy_value_fn(self, board):
        """
        评估场面局势，给出每个位置的概率及价值
        Args:
            input: 棋盘状态
            output: 返回一组列表，包含棋盘每个可下的点的动作概率以及价值得分。
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(
            board.current_state().reshape(
                -1, 4, self.board_width, self.board_height
            )
        ).astype("float32")

        act_probs, value = self.policy_value_evaluate(current_state)
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value

# CORRECT VERSION
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """用采样得到的样本集合对策略价值网络进行一次训练"""
    # wrap in Tensor
        state_batch = torch.from_numpy(state_batch).to(self.device)
        mcts_probs = torch.from_numpy(mcts_probs).to(self.device)
        winner_batch = torch.from_numpy(winner_batch).to(self.device)

    # zero the parameter gradients
        self.optimizer.zero_grad()
    # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # forward
        self.policy_value_net.train()
        log_act_probs, value = self.policy_value_net(state_batch)
    
    # define the loss = (z - v)^2 - pi^T * log(p)
        value = value.view(-1)
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -torch.mean(
            torch.sum(mcts_probs * log_act_probs, dim=1)
        )
        loss = value_loss + policy_loss
    
    # backward and optimize
        loss.backward()
        self.optimizer.step()
    
    # Return all three loss components as a tuple
        return loss.item(), value_loss.item(), policy_loss.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """保存模型"""
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def policy_value_fn(board):
    """
    接受状态并输出（动作，概率）列表的函数元组和状态的分数"""
    # 返回统一概率和0分的纯MCTS
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0

class TreeNode(object):
    """MCTS树中的节点。

    每个节点跟踪其自身的值Q，先验概率P及其访问次数调整的先前得分u。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 从动作到TreeNode的映射
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """通过创建新子项来展开树。
        action_priors：一系列动作元组及其先验概率根据策略函数.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在子节点中选择能够提供最大行动价值Q的行动加上奖金u（P）。
        return：（action，next_node）的元组
        """
        return max(
            self._children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct),
        )

    def update(self, leaf_value):
        """从叶节点评估中更新节点值
        leaf_value: 这个子树的评估值来自从当前玩家的视角
        """
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值,取对于所有访问次数的平均数
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """就像调用update（）一样，但是对所有祖先进行递归应用。"""
        # 如果它不是根节点，则应首先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值。它是叶评估Q和此节点的先验的组合
        调整了访问次数，u。
        c_puct：控制相对影响的（0，inf）中的数字，该节点得分的值Q和先验概率P.
        """
        self._u = (
            c_puct
            * self._P
            * np.sqrt(self._parent._n_visits)
            / (1 + self._n_visits)
        )
        return self._Q + self._u

    def is_leaf(self):
        """检查叶节点（即没有扩展的节点）。"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """对蒙特卡罗树搜索的一个简单实现"""

    def __init__(
        self, policy_value_fn, c_puct=5, n_playout=10000, mode="train"
    ):
        """
        policy_value_fn：一个接收板状态和输出的函数（动作，概率）元组列表以及[-1,1]中的分数
             （即来自当前的最终比赛得分的预期值玩家的观点）对于当前的玩家。
        c_puct：（0，inf）中的数字，用于控制探索的速度收敛于最大值政策。 更高的价值意味着
                 依靠先前的更多。
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.mode = mode

    def _playout(self, state):
        """从根到叶子运行单个播出，获取值
        叶子并通过它的父母传播回来。
        State已就地修改，因此必须提供副本。
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # 贪心算法选择下一步行动
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 使用网络评估叶子，该网络输出（动作，概率）元组p的列表以及当前玩家的[-1,1]中的分数v。
        action_probs, leaf_value = self._policy(state)
        # 查看游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        if self.mode == "train":
            if end:
                # 对于结束状态,将叶子节点的值换成"true"
                if winner == -1:  # tie
                    leaf_value = 0.0
                else:
                    leaf_value = (
                        1.0 if winner == state.get_current_player() else -1.0
                    )
        else:
            # 通过随机的rollout评估叶子结点
            leaf_value = self._evaluate_rollout(state)
        # 在本次遍历中更新节点的值和访问次数
        # Note: leaf_value is from the perspective of the player at the leaf node
        # but update_recursive expects it from the perspective of the player
        # who just made the move to reach this leaf node. So we negate it.
        node.update_recursive(-leaf_value)

    @staticmethod
    def _evaluate_rollout(state, limit=1000):
        """使用推出策略直到游戏结束，
        如果当前玩家获胜则返回+1，如果对手获胜则返回-1，
        如果是平局则为0。
        """
        player = state.get_current_player()
        winner = -1
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            # In evaluation mode, we might not have a board instance available
            # globally, so we access availables through the state object.
            if len(state.availables) == 0:
                break
            max_action = random.choice(state.availables)
            state.do_move(max_action)
        else:
            # 如果没有从循环中断，请发出警告。
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state, temp=1e-3):
        """
        如果 prob 为 True，则按顺序运行所有播出并返回可用的操作及其相应的概率。
        否则按顺序运行所有播出并返回访问量最大的操作。
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        if self.mode == "train":
            # 根据根节点处的访问计数来计算移动概率
            act_visits = [
                (act, node._n_visits)
                for act, node in self._root._children.items()
            ]
            acts, visits = zip(*act_visits)
            act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

            return acts, act_probs

        return max(
            self._root._children.items(),
            key=lambda act_node: act_node[1]._n_visits,
        )[0]

    def update_with_move(self, last_move):
        """保留我们已经知道的关于子树的信息"""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """基于MCTS的AI玩家"""

    def __init__(
        self,
        policy_value_function=policy_value_fn,
        c_puct=5,
        n_playout=2000,
        is_selfplay=0,
        mode="train",
    ):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, mode)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # 像alphaGo Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            if self.mcts.mode == "train":
                acts, probs = self.mcts.get_move(board, temp)
                move_probs[list(acts)] = probs
                if self._is_selfplay:
                    # 添加Dirichlet Noise进行探索（自我训练所需）
                    move = np.random.choice(
                        acts,
                        p=0.75 * probs
                        + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))),
                    )
                    # 更新根节点并重用搜索树
                    self.mcts.update_with_move(move)
                else:
                    # 使用默认的temp = 1e-3，它几乎相当于选择具有最高概率的移动
                    move = np.random.choice(acts, p=probs)
                    # 重置根节点
                    self.mcts.update_with_move(-1)

                if return_prob:
                    return move, move_probs
                else:
                    return move
            else:
                move = self.mcts.get_move(board)
                self.mcts.update_with_move(-1)
                return move
        else:
            print("棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)


#  对于五子棋的AlphaZero的训练的实现
class TrainPipeline:
# CORRECTED __init__ method in TrainPipeline

    def __init__(self, init_model=None, file_path="test", use_gpu=True):
    # Basic game and training parameters
        self.board_width = 9
        self.board_height = 9
        self.n_in_row = 5
        self.board = Board(
            width=self.board_width,
            height=self.board_height,
            n_in_row=self.n_in_row,
        )
        self.game = Game_UI(self.board)
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 200
        self.game_batch_num = 1000 # Increased for longer training
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000

    # CORRECT ORDER: Create the policy_value_net FIRST
        if init_model:
        # From an existing model file
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height,
                model_file=init_model,
                use_gpu=use_gpu,
            )
        else:
        # Start training from scratch
            self.policy_value_net = PolicyValueNet(
                self.board_width, self.board_height, use_gpu=use_gpu
            )

    # THEN, create the mcts_player which DEPENDS on the policy_value_net
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
        )

    # The rest of the initializations
        self.file_path = file_path
        self.episode_len = 0

    # Directory and CSV log setup
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        self.log_file_path = os.path.join(self.file_path, 'training_log1.csv')
        self.log_headers = [
            'batch_i', 'episode_len', 'loss', 'value_loss', 'policy_loss',
            'kl_divergence', 'lr_multiplier', 'win_ratio',
            'win_count', 'loss_count', 'tie_count'
        ]
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.log_headers)
                writer.writeheader()

    def get_equi_data(self, play_data):
        """通过旋转和翻转来增加数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(
                        mcts_porb.reshape(self.board_height, self.board_width)
                    ),
                    i,
                )
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner)
                )
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner)
                )
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """收集自我博弈数据进行训练"""
        for i in range(n_games):
            winner, play_data = self.game.start_play_train(
                self.mcts_player, temp=self.temp
            )
            play_data = list(play_data)
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def update_policy_value_net(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype("float32")

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype("float32")

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype("float32")

        old_probs, old_v = self.policy_value_net.policy_value_evaluate(
            state_batch
        )
        loss = kl = 0
        for i in range(self.epochs):
            loss, v_loss, p_loss = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier,
            )
            total_loss, value_loss, policy_loss = loss, v_loss, p_loss
            new_probs, new_v = self.policy_value_net.policy_value_evaluate(
                state_batch
            )
            kl = np.mean(
                np.sum(
                    old_probs
                    * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # 自适应调节学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        return total_loss, value_loss, policy_loss, kl

    def evaluate_policy_value_net(self, n_games=10):
        """
        通过与纯的MCTS算法对抗来评估训练的策略
        注意：这仅用于监控训练进度
        """
        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
        )
        pure_mcts_player = MCTSPlayer(
            policy_value_function=policy_value_fn,
            c_puct=5,
            n_playout=self.pure_mcts_playout_num,
            mode="eval",
        )
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play_evaluate(
                current_mcts_player, pure_mcts_player, start_player=i % 2
            )
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print(
            "num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]
            )
        )
        return win_ratio, win_cnt

    def run(self):
        root = os.getcwd()
        dst_path = self.file_path
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        try:
            for i in range(self.game_batch_num):
                # === MODIFICATION START: Initialize log data for current step ===
                log_data = {key: None for key in self.log_headers}
                log_data['batch_i'] = i + 1
                # === MODIFICATION END ===

                self.collect_selfplay_data(self.play_batch_size)
                log_data['episode_len'] = self.episode_len
                print(f"batch i:{i+1}, episode_len:{self.episode_len}")

                if len(self.data_buffer) > self.batch_size:
                    loss, v_loss, p_loss, kl = self.update_policy_value_net()
                    # === MODIFICATION START: Store log data ===
                    log_data['loss'] = loss
                    log_data['value_loss'] = v_loss
                    log_data['policy_loss'] = p_loss
                    log_data['kl_divergence'] = kl
                    log_data['lr_multiplier'] = self.lr_multiplier
                    print(f"loss: {loss:.4f}, value_loss: {v_loss:.4f}, policy_loss: {p_loss:.4f}, kl: {kl:.4f}, lr_mult: {self.lr_multiplier:.2f}")
                    # === MODIFICATION END ===

                if (i + 1) % self.check_freq == 0:
                    print(f"current self-play batch: {i+1}")
                    # === MODIFICATION START: Get full evaluation results ===
                    win_ratio, win_cnt = self.evaluate_policy_value_net()
                    log_data['win_ratio'] = win_ratio
                    log_data['win_count'] = win_cnt[1]
                    log_data['loss_count'] = win_cnt[2]
                    log_data['tie_count'] = win_cnt[-1]
                    # === MODIFICATION END ===
                    self.policy_value_net.save_model(
                        os.path.join(dst_path, 'current_policy.model')
                    )
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(os.path.join(dst_path, 'best_policy.model'))
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 8000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                
                # === MODIFICATION START: Write log data to CSV ===
                with open(self.log_file_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.log_headers)
                    writer.writerow(log_data)
                # === MODIFICATION END ===

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == "__main__":
    # model_path = 'model_ygh/best_policy.model'
    model_path = 'test/current_policy.model'
    
    use_gpu_available = torch.cuda.is_available()

    # To load an existing model, uncomment the following line
    training_pipeline = TrainPipeline(init_model=None, use_gpu=use_gpu_available)
    #training_pipeline = TrainPipeline(None, use_gpu=use_gpu_available)
    loss_list = training_pipeline.run()