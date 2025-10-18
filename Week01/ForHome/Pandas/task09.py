import pandas as pd

def main() -> None:
    df_cars = pd.read_csv("D:\\UNIVERSITY\\ML\\Week01\\ForHome\\Data\\cars_advanced.csv")
    df_cars.set_index('Unnamed: 0')
    print(f'Before:\n{df_cars}')

    df_cars['COUNTRY'] = df_cars['country'].apply(str.upper)
    print(f'\nAfter:\n{df_cars}')

if __name__ == "__main__":
    main()