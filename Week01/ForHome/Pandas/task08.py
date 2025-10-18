import pandas as pd

def main() -> None:
    df_cars = pd.read_csv("D:\\UNIVERSITY\\ML\\Week01\\ForHome\\Data\\cars_advanced.csv", index_col=0)
    print(f'Before:\n{df_cars}')

    df_cars['COUNTRY'] = 'NAN'
    for index, row in df_cars.iterrows():
        row['COUNTRY'] = row['country'].upper()
    print(f'\nAfter:\n{df_cars}')

if __name__ == "__main__":
    main()