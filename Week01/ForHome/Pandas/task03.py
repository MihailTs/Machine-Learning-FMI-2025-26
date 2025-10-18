import pandas as pd

def main() -> None:
    df_cars = pd.read_csv("D:\\UNIVERSITY\\ML\\Week01\\ForHome\\Data\\cars_advanced.csv")
    print(df_cars)
    df_cars.set_index(df_cars.columns[0], inplace=True)
    df_cars.index.name = None
    print('\nAfter setting first column as index')
    print(df_cars)

if __name__ == "__main__":
    main()