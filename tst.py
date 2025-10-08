import pandas as pd

data = pd.read_csv("clockify_full_report.csv")

# Make column names lowercase
data.columns = data.columns.str.lower()

# Make all string values lowercase
data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)
data['start date'] = pd.to_datetime(data['start date'])
data['end date'] = pd.to_datetime(data['end date'])

if 'description' in data.columns and 'task' in data.columns:
        data['task'] = data.apply(
            lambda row: row['description'] if pd.isna(row['task']) or row['task'] == '' else row['task'],
            axis=1
        )
print(data.head())