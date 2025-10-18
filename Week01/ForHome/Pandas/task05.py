import pandas as pd

def main() -> None:
    df_cars = pd.read_csv("D:\\UNIVERSITY\\ML\\Week01\\ForHome\\Data\\cars_advanced.csv", index_col=0)
    print(f'{df_cars[df_cars['drives_right'] == True]}\n')
    print(f'{df_cars[df_cars['cars_per_cap'] > 500]['country']}\n')
    print(f'{df_cars[(df_cars['cars_per_cap'] > 10) & (df_cars['cars_per_cap'] < 80)]['country']}\n')
    print('Alternative')
    print(df_cars[df_cars['cars_per_cap'].between(10, 80)]['country'])

if __name__ == "__main__":
    main()