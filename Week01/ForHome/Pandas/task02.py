import pandas as pd

def main() -> None:
    names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
    dr =  [True, False, False, False, True, True, True]
    cpc = [809, 731, 588, 18, 200, 70, 45]

    df_driving = pd.DataFrame({'country':names, 'drives_right':dr, 'cars_per_capita':cpc})
    df_driving.index = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
    print(df_driving)

if __name__ == "__main__":
    main()