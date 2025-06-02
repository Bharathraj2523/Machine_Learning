import xgboost as xgb
import pandas as pd 

def predict_col(input_df, rawdf, numerical_cols , cat_col, target):

    from sklearn.preprocessing import StandardScaler

    scale_col = numerical_cols + cat_col + ['area' , 'power']
    scale_col.remove(target)

    scaler = StandardScaler().fit(rawdf[scale_col])

    input_df_copy = input_df.copy()
    input_df_copy[scale_col] = scaler.transform(input_df_copy[scale_col])

    known_df = input_df_copy[input_df_copy[target].notna()]
    unknown_df = input_df_copy[input_df_copy[target].isna()]

    feature = numerical_cols + cat_col +['area' , 'power']
    if target in feature:
        feature.remove(target)

    known_df = known_df.dropna(subset=feature)
    unknown_df = unknown_df.dropna(subset=feature)

    x_train = known_df[feature]
    y_train = known_df[target]
    x_test = unknown_df[feature]

    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    model.fit(x_train, y_train)

    predicted_column = model.predict(x_test)

    unknown_df = unknown_df.copy()
    unknown_df[target] = predicted_column
    unknown_df.reset_index(inplace=True)
    unknown_df.rename(columns={'index': 'id'}, inplace=True)

    predicted_column_series = pd.Series(predicted_column, index=unknown_df['id'])

    rawdf.loc[rawdf[target].isna(), target] = (
        rawdf.loc[rawdf[target].isna(), 'id'].map(predicted_column_series)
    )

    return rawdf


def fill_voltage_current_power_irradiance(new_df):
    # Step 1: Compute power where irradiance, area, and efficiency are known
    power_mask = (
        new_df['irradiance'].notna() & 
        new_df['area'].notna() & 
        new_df['efficiency'].notna()
    )
    new_df.loc[power_mask, 'power'] = (
        new_df.loc[power_mask, 'irradiance'] *
        new_df.loc[power_mask, 'area'] *
        new_df.loc[power_mask, 'efficiency']
    )

    # Step 2: Impute voltage (if power and current are known)
    missing_voltage = (
        new_df['voltage'].isna() &
        new_df['power'].notna() &
        new_df['current'].notna()
    )
    new_df.loc[missing_voltage, 'voltage'] = (
        new_df.loc[missing_voltage, 'power'] /
        new_df.loc[missing_voltage, 'current']
    )

    # Step 3: Impute current (if power and voltage are known)
    missing_current = (
        new_df['current'].isna() &
        new_df['power'].notna() &
        new_df['voltage'].notna()
    )
    new_df.loc[missing_current, 'current'] = (
        new_df.loc[missing_current, 'power'] /
        new_df.loc[missing_current, 'voltage']
    )

    # Step 4: Impute irradiance (if voltage, current, area, efficiency are known)
    missing_irradiance = (
        new_df['irradiance'].isna() &
        new_df['voltage'].notna() &
        new_df['current'].notna() &
        new_df['area'].notna() &
        new_df['efficiency'].notna()
    )
    new_df.loc[missing_irradiance, 'irradiance'] = (
        (new_df.loc[missing_irradiance, 'voltage'] * new_df.loc[missing_irradiance, 'current']) /
        (new_df.loc[missing_irradiance, 'area'] * new_df.loc[missing_irradiance, 'efficiency'])
    )

    return new_df
