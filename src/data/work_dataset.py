# dvc stage add -n preprocess -d data/raw/life_expectancy_data.csv -d src/data/work_dataset.py  -o data/EDA_processed/life_expectancy_data_processed.csv  -o data/EDA_processed/data_x.npy -o data/EDA_processed/data_y.npy  -o data/EDA_processed/test_data.csv -o data/EDA_processed/train_data.csv python src/data/work_dataset.py
import pandas as pd
import numpy as np
import yaml

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

def main():
    # Первый этап. Чтение файла
    with open('C:/ML_Labs/LB1_2_ML_Yakushev/config/parameters.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    
    df = pd.read_csv(config['data']['dataset_csv'])

    # Второй этап. Обработка данных
    df.drop('Country', axis=1, inplace=True)

    lb = LabelBinarizer()
    df['Status'] = lb.fit_transform(df['Status'])

    df.drop(['Year', 'Total expenditure', 'Population', 'Measles', 
         'Hepatitis B', 'under-five deaths', 'infant deaths'],
         axis=1, inplace=True)
    
    df.drop(['percentage expenditure', 'thinness 5-9 years', 
         'Income composition of resources'], axis=1, inplace=True)

    df.dropna(inplace=True)

    # Третий этап. Сохранение данных
    df.to_csv(config['data']['new_dataset_path'], index=False)

    data_x = np.array(df.drop(config['data']['target'], axis=1))
    data_y = np.array(df[config['data']['target']])
    
    np.save(config['data']['dataset_x_path'], data_x)
    np.save(config['data']['dataset_y_path'], data_y)
    
    train_df, test_df = train_test_split(
        df,
        test_size=config['base']['test_size'],
        random_state=config['base']['random_state']
    )

    train_df.to_csv(config['data']['train_path'], index=False)
    test_df.to_csv(config['data']['test_path'], index=False)

    return 0

if __name__ == '__main__':
    main()