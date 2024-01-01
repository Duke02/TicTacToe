import typing as tp 
import numpy as np

PlayerId = int
PlayerLabel = str
AiMoveChoiceFunc = tp.Callable[[np.ndarray], tp.Tuple[int, int]]

num_to_player: tp.Dict[PlayerId, PlayerLabel] = {0: '_', 1: 'X', -1: 'O'}
player_to_num: tp.Dict[PlayerLabel, PlayerId] = {p: n for n, p in num_to_player.items()}


def get_cell(state: np.ndarray, x: int, y: int) -> PlayerId:
    return state[x, y]


def get_cell_label(state: np.ndarray, x: int, y: int) -> PlayerLabel:
    return num_to_player[get_cell(state, x, y)] 


def is_empty_cell(state: np.ndarray, x: int, y: int) -> bool:
    return get_cell(state, x, y) == 0


def set_cell(state: np.ndarray, x: int, y: int, num: PlayerId) -> np.ndarray:
    if not is_empty_cell(state, x, y):
        raise ValueError(f'Trying to set a value of {num} to a non empty cell at ({x}, {y}).')
    out: np.ndarray = state.copy()
    out[x, y] = num
    return out 


def set_cell_label(state: np.ndarray, x: int, y: int, label: PlayerLabel) -> np.ndarray:
    return set_cell(state, x, y, player_to_num[label])


def setup_state(state_size: int) -> np.ndarray:
    return np.zeros((state_size, state_size)).astype(int)


def get_state_size_from_difficulty(difficulty: int) -> int:
    return difficulty_settings[difficulty][0]


def get_ai_move_func_from_difficulty(difficulty: int) -> AiMoveChoiceFunc:
    return difficulty_settings[difficulty][1]


def get_state_size(state: np.ndarray) -> int:
    return state.shape[0]


def get_winner(state: np.ndarray) -> tp.Optional[int]:
    dim0_sums: np.ndarray = np.sum(state, axis=0)
    state_size: int = get_state_size(state)
    winner_idx: np.ndarray = np.argwhere(np.abs(dim0_sums) == state_size)
    if winner_idx.size > 0:
        return np.sign(dim0_sums[winner_idx[0]])

    dim1_sums: np.ndarray = np.sum(state, axis=1)
    winner_idx: np.ndarray = np.argwhere(np.abs(dim1_sums) == state_size)
    if winner_idx.size > 0:
        return np.sign(dim1_sums[winner_idx[0]])

    diag1_sums: int = np.sum(state.diagonal())
    if np.abs(diag1_sums) == state_size:
        return np.sign(diag1_sums)

    diagT_indices: np.ndarray = np.array([(i, state_size - i - 1) for i in range(state_size)])
    diagT_sums: int = np.sum(state[diagT_indices[:, 0], diagT_indices[:, 1]])
    if np.abs(diagT_sums) == state_size:
        return np.sign(diagT_sums)

    # No one has won, so either the game is not done or the board is full
    # and it's a tie. 
    return None


def is_done(state: np.ndarray) -> bool:
    return np.all(state != 0) or get_winner(state) is not None 


def print_state(state: np.ndarray) -> tp.NoReturn:
    for y in range(get_state_size(state)):
        print(' | '.join([get_cell_label(state, x, y) for x in range(get_state_size(state))]))


def get_random_xy(state: np.ndarray) -> tp.Tuple[int, int]:
    possible_coords: np.ndarray = np.argwhere(state == 0)
    np.random.shuffle(possible_coords)
    return tuple(possible_coords[0].tolist())


def is_valid_input(state: np.ndarray, x: int, y: int) -> bool:
    return get_cell(state, x, y) == 0


def get_possible_moves(state: np.ndarray) -> tp.List[tp.Tuple[int, int]]:
    possible_moves: np.ndarray = np.argwhere(state == 0)
    return [(xy[0], xy[1]) for xy in possible_moves]


def get_denominator(state_size: int) -> float:
    return state_size ** 4.0


def get_heuristic_of_done_game(state: np.ndarray) -> float:
    winner: tp.Optional[int] = get_winner(state)
    state_size: int = get_state_size(state)
    denom: int = get_denominator(state_size) 
    if winner is None:
        # If we have a tie, then give a less bad penalty than a straight up loss. 
        return -0.5
    elif winner < 0:
        # If the computer wins, then give a good reward!
        return 1
    else:
        # If the player wins, then give a stronger penalty.
        return -1


def get_heuristic_of_0depth(state: np.ndarray) -> float:
    state_size: int = get_state_size(state)
    sum_rows: np.ndarray = np.sum(state, axis=0)
    sum_cols: np.ndarray = np.sum(state, axis=1)
    sum_diag: int = np.sum(np.diagonal(state))
    
    diagT_indices: np.ndarray = np.array([(i, state_size - i - 1) for i in range(state_size)])
    sum_inv_diag: int = np.sum(state[diagT_indices[:, 0], diagT_indices[:, 1]])
    # If we run out of depth to explore the game tree,
    # then sum up each row and column and diagonal that
    # the computer is more likely to win.
    return ((np.sum(sum_rows * ((sum_rows < 0).astype(float) - (sum_rows > 0).astype(float)))) + (np.sum(sum_cols * ((sum_cols < 0).astype(float) - (sum_cols > 0).astype(float)))) + (-sum_diag) + (-sum_inv_diag)) / get_denominator(state_size)



def get_infinity(state: np.ndarray) -> float:
    '''
    Get a reasonably large enough value for infinity. It's based off of the state_size.
    '''
    return 2 * ((5.0 * get_state_size(state) + 2) ** 3)


def alpha_beta(state: np.ndarray, x: int, y: int, depth: int, alpha: float, beta: float, maximizing_player: bool) -> tp.Tuple[float, int, int]:
    if is_done(state):
        return get_heuristic_of_done_game(state), x, y
    elif depth == 0:
        return get_heuristic_of_0depth(state), x, y

    infinity: int = get_infinity(state)

    if maximizing_player:
        value: float = -infinity
        best_x: int = x
        best_y: int = y
        for move_x, move_y in get_possible_moves(state):
            found_value, found_x, found_y = alpha_beta(set_cell(state, move_x, move_y, -1), move_x, move_y, depth - 1, alpha, beta, False)
            if found_value > value:
                best_x, best_y = found_x, found_y
                value = found_value 
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value, best_x, best_y 
    else:
        value: float = infinity
        best_x: int = x
        best_y: int = y
        for move_x, move_y in get_possible_moves(state):
            found_value, found_x, found_y = alpha_beta(set_cell(state, move_x, move_y, 1), move_x, move_y, depth - 1, alpha, beta, True)
            if found_value < value:
                best_x, best_y = found_x, found_y
                value = found_value
            beta = min(beta, value)
            if value <= alpha:
                break
        return value, best_x, best_y


def do_smart_ai(state: np.ndarray) -> tp.Tuple[int, int]:
    infinity: int = get_infinity(state)
    init_x, init_y = get_random_xy(state)
    out_value, out_x, out_y = alpha_beta(state, init_x, init_y, get_state_size(state), -infinity, infinity, True)
    assert is_valid_input(state, out_x, out_y), 'AI suggested a move that is an invalid input'
    print(f'Changed initial prediction of ({init_x}, {init_y}) that ({out_x}, {out_y}) would give a value of {out_value}')
    return out_x, out_y


def do_dumb_smart_ai(state: np.ndarray) -> tp.Tuple[int, int]:
    if np.random.random() < 0.5:
        return get_random_xy(state)
    else:
        return do_smart_ai(state)


def get_smarter_ai(state: np.ndarray) -> tp.Tuple[int, int]:
    round_number: int = np.sum(np.abs(state)) // 2
    state_size: int = get_state_size(state)
    if np.random.random() < round_number / (2 * state_size):
        return do_smart_ai(state)
    else:
        return get_random_xy(state)


def get_difficulty_from_user() -> int:
    print('Choose your difficulty: ')
    for difficulty_id, (_, _, difficulty_name) in difficulty_settings.items():
        print(f'[{difficulty_id}] :: {difficulty_name}')
    return int(input('Choose the difficulty by its number> '))


difficulty_settings: tp.Dict[int, tp.Tuple[int, AiMoveChoiceFunc, str]] = {0: (3, get_random_xy, 'easy'), 1: (3, do_dumb_smart_ai, 'less easy'), 2: (3, do_smart_ai, 'medium'), 3: (5, get_smarter_ai, 'almost hard'), 4: (5, do_smart_ai, 'hard'), 50: (100, get_random_xy, 'near impossible'), 75: (100, do_dumb_smart_ai, 'almost impossible'), 100: (100, do_smart_ai, 'impossible')}


def do_game_loop(difficulty: int) -> int:
    # Assume the player chooses X as their label.
    is_players_turn: bool = False 

    board_size: int = get_state_size_from_difficulty(difficulty)
    board_state: np.ndarray = setup_state(board_size)

    round_num: int = 1

    while not is_done(board_state):
        if is_players_turn:
            print(f'=== ROUND {round_num} ===')
            print_state(board_state)
            is_good_input: bool = False
            while not is_good_input:
                x: int = int(input(f'What is your X coordinate (0 -> {board_size - 1})'))
                y: int = int(input(f'What is your Y coordinate (0 -> {board_size - 1})'))
                if is_valid_input(board_state, x, y):
                    is_good_input = True 
                    break
                else:
                    print(f'Invalid Input. Please try again.')
            
            board_state = set_cell(board_state, x, y, 1)
            is_players_turn = False
        else:
            x, y = get_ai_move_func_from_difficulty(difficulty)(board_state)
            board_state = set_cell(board_state, x, y, -1)
            is_players_turn = True 
            round_num += 1
    print('=== GAME OVER ===')
    print(f'Board after {round_num} rounds: ')
    print_state(board_state)

    winner: tp.Optional[int] = get_winner(board_state)

    if winner is None or winner == 0:
        print('It\'s a tie!')
        return 0
    elif winner < 0:
        print('Computer won!')
    else:
        print('Player won!')

    return winner


if __name__ == '__main__':
    difficulty: int = get_difficulty_from_user()
    do_game_loop(difficulty)

