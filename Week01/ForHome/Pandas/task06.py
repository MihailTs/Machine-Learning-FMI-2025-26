import pandas as pd

def main() -> None:
    df_cars = pd.read_csv("D:\\UNIVERSITY\\ML\\Week01\\ForHome\\Data\\cars_advanced.csv", index_col=0)
    for index, row in df_cars.iterrows():
        print(f'Label is \"{index}\"')
        print('Row contents:')
        print(f'{pd.Series(row)}\n')

if __name__ == "__main__":
    main()