def impute_median(df, column_name):
    """
    This function will impute string 'NA' with the median value
    :param df:
    :param column_name:
    :return:
    """
    # cleaning for Customer seniority
    median_value = df[~df[column_name].str.contains('NA')][column_name].astype(float).astype(int).median()
    df.loc[df[column_name].str.contains('NA'), column_name] = median_value
    df[column_name] = df[column_name].astype(float).astype(int)
    return df
