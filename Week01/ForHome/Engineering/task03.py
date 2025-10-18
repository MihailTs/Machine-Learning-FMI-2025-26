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
            level = max(level - 1, 0)
        elif dice_roll < 6:
            level = level + 1
        else:
            dice_roll = rng.integers(1, 7)
            level = level + dice_roll
        results[i] = level
    
    print(results)

if __name__ == "__main__":
    main()