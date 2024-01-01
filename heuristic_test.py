import typing as tp
import numpy as np
from main import get_heuristic_of_done_game, get_heuristic_of_0depth
from tqdm import tqdm 


Computer: int = -1
Human: int = 1
Empty: int = 0


def get_initial_state(state_size: int = 3) -> np.ndarray:
    return np.zeros((state_size, state_size))


def check_signs() -> bool:
    state_size: int = 3
    state1: np.ndarray = get_initial_state(state_size)
    state1[1, :] = Computer
    assert get_heuristic_of_done_game(state1) > 0, 'Done game where computer wins does not give a positive heuristic'

    state1[1, :] = Human
    assert get_heuristic_of_done_game(state1) < 0, 'Done game where human player wins does not give a negative heuristic'

    state2: np.ndarray = get_initial_state(state_size)
    state2[1, :-1] = Computer
    assert get_heuristic_of_0depth(state2) > 0, 'Not Finished game where Computer player has the advantage does not give a positive heuristic'

    state2[1, :-1] = Human
    assert get_heuristic_of_0depth(state2) < 0, 'Not Finished game where Human player has the advantage does not give a negative heuristic'
    
    return True


def check_mixed_signs() -> bool:
    state_size: int = 3
    state1: np.ndarray = get_initial_state(state_size)
    state1[2, :-1] = Computer
    state1[0, :-1] = Human
    value1: int = get_heuristic_of_0depth(state1)
    print(f'\t\tTest 1: Computer == Human - Value: {value1}')
    assert value1 == 0, f'Mixed game where Computer and Human are tied resulted in a non-zero number {value1}'

    state2: np.ndarray = get_initial_state(state_size)
    state2[1, :-1] = Computer
    state2[0, 0] = Human
    value2: int = get_heuristic_of_0depth(state2)
    print(f'\t\tTest 2: Computer > Human - Value: {value2}')
    assert value2 > 0, f'Mixed game where Computer had an advantage against the Human player resulted in a non-positive number {value2}'

    state3: np.ndarray = get_initial_state(state_size)
    state3[1, 0] = Computer
    state3[0, :-1] = Human
    value3: int = get_heuristic_of_0depth(state3)
    print(f'\t\tTest 3: Computer < Human - Value: {value3}')
    assert value3 < 0, f'Mixed game where Computer had an disadvantage against the Human player resulted in a non-negative number {value3}'
    
    return True


def check_relations() -> bool:
    state_size: int = 3
    state1a: np.ndarray = get_initial_state(state_size)
    state1b: np.ndarray = get_initial_state(state_size)
    state1a[1, :-1] = Computer
    state1a[0, 0] = Human
    state1b[1, :-1] = Human
    state1b[0, 0] = Computer
    value1a: int = get_heuristic_of_0depth(state1a)
    value1b: int = get_heuristic_of_0depth(state1b)
    print(f'\t\tTest 1: A == -B - Value A: {value1a}, Value B: {value1b}')
    assert value1a > value1b, 'Comparison between two opposite games where Game A had the computer winning with Game B having the Human Player winning did not result in Game A being greater than Game B.'
    assert abs(value1a) == abs(value1b), f'Comparison between two opposite games where Game A had the Computer winning with Game B having the Human Player winning did not give the same magnitude of scores'

    state2a: np.ndarray = get_initial_state(state_size)
    state2b: np.ndarray = get_initial_state(state_size)
    state2a[1, :-1] = Computer
    state2a[0, 0] = Human
    state2b[:-1, 1] = Computer
    state2b[0, 0] = Human
    value2a: int = get_heuristic_of_0depth(state2a)
    value2b: int = get_heuristic_of_0depth(state2b)
    print(f'\t\tTest 2: A == B.T - Value A: {value2a}, Value B: {value2b}')
    assert value2a == value2b, 'Comparison between two games where Game A had the computer winning in columns with Game B having the computer winning in rows did not result in Game A being equal to Game B.'
    
    state3a: np.ndarray = get_initial_state(state_size)
    state3b: np.ndarray = get_initial_state(state_size)
    state3a[1, :-1] = Computer
    state3a[0, 0] = Human
    state3b[[(i, i) for i in range(state_size)]] = Computer
    state3b[1, 0] = Human
    value3a: int = get_heuristic_of_0depth(state3a)
    value3b: int = get_heuristic_of_0depth(state3b)
    print(f'\t\tTest 3: A(row) < B(diag) - Value A: {value3a}, Value B: {value3b}')
    assert value3a < value3b, 'Comparison between two games where Game A had the computer winning in the rows and Game B had the computer winning in the diagonals resulted in Game B not having a better value than Game A. (due to there being more opportunities to win for Game B)'

    return True 


def run_test_func(test_func: tp.Callable[[], bool], do_printing: bool = True) -> bool:
    if do_printing:
        print(f'Testing {test_func.__name__}...')
    try:
        test_func()
        if do_printing:
            print(f'\t{test_func.__name__} passed!')
        return True
    except AssertionError as e:
        print(f'\tGot an assertion error {e}')
        return False


if __name__ == '__main__':
    print('Testing heuristic functions...')
    test_functions: tp.List[tp.Callable[[], tp.NoReturn]] = [check_signs, check_mixed_signs, check_relations]

    num_failed: int = 0

    for f in test_functions:
        succeeded: bool = run_test_func(f) 
        if not succeeded:
            num_failed += 1
        print(' ')

    if num_failed == 0:
        print('All tests passed!')
    else:
        print(f'{num_failed} tests failed.')



