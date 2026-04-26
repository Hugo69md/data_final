import pandas as pd

df = pd.read_csv('data.csv')

# Remove double headers
for i in df.columns:
    df = df[df[i] != i]

# remove " " from column "Country Name""
df['Country Name'] = df['Country Name'].str.replace(' ', '')

#use regex in column 3 to extract year values and keep only the year values
df['Year'] = df['Year'].str.extract(r'(\d{4})')

#remove values for line 15 in columns 3 to 11
for i in df.columns[3:11]:
    df.loc[14, i] = "NaN"
    df.loc[15, i] = "NaN"

# remove % signs and convert to numeric
for i in df.columns[2:11]:
    df[i] = df[i].str.replace('%', '')

# replace non-numeric values in columns 3 to 11 with "N/A"
for i in df.columns[2:11]:
    converted = pd.to_numeric(df[i], errors='coerce')
    df[i] = converted.where(converted.notna(), "NaN")


# convert columns 3 to 11 to numeric
for i in df.columns[2:11]:
    df[i] = pd.to_numeric(df[i], errors='coerce')

#keep only the first 4 letters of the country name in column 1
df['Country Name'] = df['Country Name'].str[:4]

#country names should be in lowercase
df['Country Name'] = df['Country Name'].str.lower()

print(df.info())
print(df.describe())

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_data.csv', index=False)