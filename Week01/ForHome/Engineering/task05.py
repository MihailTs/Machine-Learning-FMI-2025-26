import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=123)

def main() -> None:
    level = 0
    walks_number = 5
    reps = 100
    walks = list(range(walks_number))

    for i in range(walks_number):
        level = 0
        results = list(range(reps))
        results[0] = level

        for j in range(reps):
            dice_roll = rng.integers(1, 7)
            if dice_roll < 3:
                level = max(level - 1, 0)
            elif dice_roll < 6:
                level = level + 1
            else:
                dice_roll = rng.integers(1, 7)
                level = level + dice_roll
            results[j] = level
        walks[i] = results
    
    print(walks)

if __name__ == "__main__":
    main()