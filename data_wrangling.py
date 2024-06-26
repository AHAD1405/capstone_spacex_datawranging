import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def deal_missing(df):
    '''
        Function that find out missing value and deal with it 
    '''
    # loop over each variable
    for column in df.columns:
        missing_perc = df[column].isnull().sum()/df.shape[0]*100  # Calculate missing percentage for each variable
        if missing_perc == 0.0: continue # In case: no missing 
        if missing_perc > 50: df.drop(column, axis=1, inplace=True) # In case: most value missing, delete whole column
                
        # Impute Numeric Column
        if df[column].dtype != 'object':
            avg_val = df[column].mean()     # calculate mean of column
            df[column].fillna(avg_val)      # Deal with mean
        
        # Impute Categorical Column 
        else:
            #avg_val = df[column].astype("float").mean(axis=0)
            #df[column].replace(np.nan, avg_val, inplace=True)
            mode_value = df[column].mode()[0]
            df[column].replace('', mode_value, inplace=True)
            df[column].fillna(mode_value, inplace=True)
    return df

def main():
    # Reading data 
    data = pd.read_csv("dataset_collected.csv")
    df = pd.DataFrame(data)

    # Get type of each variaBLE 
    df.dtypes

    # MISSING VALUES
    df = deal_missing(df)

    # Claculate the number of lunches on each site
    group_lunchsite = df.groupby('LaunchSite')
    group_lunchsite = group_lunchsite['LaunchSite'].count()
    print('Calculate the number of lunches on each site:',group_lunchsite)
    # Calculate the number of occurrence of each Orbit.
    group_orbit = df.groupby('Orbit')
    group_orbit = group_orbit['Orbit'].count()
    print('Calculate the number of occurrence of each Orbit:\n', group_orbit)
    # Calculate the number of occurrence of mission outcome per orbit type.
    group_orbit_type = df.groupby(['Orbit','Outcome'])
    group_orbit_type = group_orbit_type['Orbit'].count()
    print('Calculate the number of occurrence of mission outcome per orbit type:\n', group_orbit_type)
    


if __name__ == '__main__':
  main()