import pandas as pd

def main() -> None:
    df_cars = pd.read_csv("D:\\UNIVERSITY\\ML\\Week01\\ForHome\\Data\\cars_advanced.csv", index_col=0)
    for index, row in df_cars.iterrows():
        print(f'{index}: {row['cars_per_cap']}')

if __name__ == "__main__":
    main()