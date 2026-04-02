import pandas as pd
import numpy as np
import yaml

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, MinMaxScaler
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

    target_col = config['data']['target']
    feature_cols = [col for col in df.columns if col != target_col]

    status_col = 'Status'
    other_features = [col for col in feature_cols if col != status_col]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[other_features] = scaler.fit_transform(df[other_features])


    # Третий этап. Сохранение данных
    df.to_csv(config['data']['new_dataset_path'], index=False)

    data_x = np.array(df.drop(target_col, axis=1))
    data_y = np.array(df[target_col])
    data_y = np.log1p(data_y)
    
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