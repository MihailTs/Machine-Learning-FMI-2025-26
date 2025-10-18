import numpy as np

rng = np.random.default_rng(seed=123)

def main() -> None:
    num = rng.random()
    print(f'Random float: {num}')
    print(f'Random integer 1: {rng.integers(1, 7)}')
    print(f'Random integer 2: {rng.integers(1, 7)}')

    level = 50
    print(f'Before throw step = {level}')
    dice_roll = rng.integers(1, 7)
    print(f'After throw dice = {dice_roll}')
    if dice_roll < 3:
        level = level - 1
    elif dice_roll < 6:
        level = level + 1
    else:
        dice_roll = rng.integers(1, 7)
        level = level + dice_roll

    print(f'After throw step = {level}')

if __name__ == "__main__":
    main()