import pandas as pd
import numpy as np
import matplotlib. pyplot as plt

edu = pd.read_csv("D:\\SSD_Python01012023\\edu1.csv",na_values = ':', usecols = ["TIME","GEO","Value"])
print(edu)

print(edu.head())

print(edu.tail())

print(edu.columns)

print(edu.values)

print(edu.describe())

print(edu['Value'])

print(edu[10:14])

print(edu.iloc[90:94,:])

print(edu.sample(10,random_state=23))

print(edu.sample(10,random_state=23).loc[73:59,:])

edu[edu['Value'] > 6.5].tail()

edu[edu['Value'].isnull()].head()

edu.max(axis=0)

print('Pandas max function:', edu['Value'].max())
print('Python max function:', max(edu['Value']))

s = edu['Value'] / 100
print(s.head())

s = edu['Value'].apply(np.sqrt)
s.head()

s = edu['Value'].apply(lambda d: d**2)
print(s.head())

edu['ValueNorm'] = edu['Value'] / edu['Value'].max()
print(edu.tail())

edu.drop('ValueNorm', axis=1, inplace=True)
print(edu.head())

#edu = edu.append({'TIME': 2000, 'Value': 5.00, 'GEO': 'a'}, ignore_index=True)
#print(edu.tail())

edu.drop(max(edu.index), axis=0, inplace=True)
print(edu.tail())

eduDrop = edu.dropna(how='any', subset=['Value'], axis=0)
print(eduDrop.head())

eduFilled = edu.fillna(value={'Value': 0})
print(eduFilled.head())

edu.sort_values(by='Value', ascending=False, inplace=True)
print(edu.head())

edu.sort_index(axis=0, ascending=True, inplace=True)
print (edu.head())

group = edu[['GEO', 'Value']].groupby('GEO').mean()
print(group.head())

filtered_data = edu[edu['TIME'] > 2005]
pivedu = pd.pivot_table(filtered_data, values='Value',
                        index=['GEO'], columns=['TIME'])
print(pivedu.head())

pivedu.loc[['Spain', 'Portugal'], [2006, 2011]]

pivedu = pivedu.drop(['Euro area (13 countries)',
                      'Euro area (15 countries)',
                      'Euro area (17 countries)',
                      'Euro area (18 countries)',
                      'European Union (25 countries)',
                      'European Union (27 countries)',
                      'European Union (28 countries)'
                      ], axis=0)
pivedu = pivedu.rename(
    index={'Germany (until 1990 former territory of the FRG)': 'Germany'})
pivedu = pivedu.dropna()
pivedu.rank(ascending=False, method='first').head()

totalSum = pivedu.sum(axis=1)
totalSum.rank(ascending=False, method='dense').sort_values().head()

fig = plt.figure(figsize=(12, 5))
totalSum = pivedu.sum(axis=1).sort_values(ascending=False)
totalSum.plot(kind='bar', style='b', alpha=0.4,
              title='Total Values for Country')
plt.savefig('Totalvalue_Country.png', dpi=300, bbox_inches='tight')


my_colors = ['b', 'r', 'g', 'y', 'm', 'c']
ax = pivedu.plot(kind='barh', stacked=True, color=my_colors, figsize=(12, 6))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Value_Time_Country.png', dpi=300, bbox_inches='tight')

plt.show()

