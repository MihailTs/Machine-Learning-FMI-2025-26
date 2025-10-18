import pandas as pd

def main() -> None:
    df_cars = pd.read_csv("D:\\UNIVERSITY\\ML\\Week01\\ForHome\\Data\\cars.csv", index_col=0)
    print(f'{pd.Series(df_cars['country'])}\n')
    print(f'{df_cars[['country']]}\n')
    print(f'{df_cars[['country','drives_right']]}\n')
    print(f'{df_cars.head(3)}\n')
    print(f'{df_cars.head(6).tail(3)}\n')

    df_cars_advanced = pd.read_csv("D:\\UNIVERSITY\\ML\\Week01\\ForHome\\Data\\cars_advanced.csv", index_col=0)
    print(f'{df_cars_advanced.loc[df_cars_advanced['country'] == 'Japan'].iloc[0]}\n')
    print(f'{df_cars_advanced.loc[(df_cars_advanced['country'] == 'Australia') | (df_cars_advanced['country'] == 'Egypt')]}\n')
    print(f'{df_cars_advanced.loc[df_cars_advanced['country'] == 'Morocco'][['drives_right']]}\n')
    df_country_road_side = df_cars_advanced[['country', 'drives_right']]
    print(f'{df_country_road_side.loc[(df_country_road_side['country'] == 'Russia') | (df_country_road_side['country'] == 'Morocco')]}\n')

if __name__ == "__main__":
    main()