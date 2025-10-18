import numpy as np

rng = np.random.default_rng(seed=123)

def main() -> None:
    level = 0
    reps = 100
    results = list(range(reps))
    results[0] = level
    for i in range(reps):
        dice_roll = rng.integers(1, 7)
        if dice_roll < 3:
            level = level - 1
        elif dice_roll < 6:
            level = level + 1
        else:
            dice_roll = rng.integers(1, 7)
            level = level + dice_roll
        results[i] = level
    
    print(results)

    # Not something very unexpected, but the level is mostly
    # greater than the starting one, which is to be expacted
    # since the expected value of every repetition is positive
    # its strange that we are on a negative step😄

if __name__ == "__main__":
    main()