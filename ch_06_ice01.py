import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

ice = pd.read_csv("D:\SSD_Python01012023\SeaIce.txt",
delim_whitespace= True)
print ('shape:', ice.shape)
print(ice.head() )

print(ice.describe() )

# Visualize the data
x = ice.year
y = ice.extent
plt.scatter(x, y, color = 'red')
plt.xlabel('Year')
plt.ylabel('Extent')
plt.show()

print('Different values in data_type field:', np.unique(ice.data_type.values))   # there is a -9999 value!

print(ice[(ice.data_type != 'Goddard')
          & (ice.data_type != 'NRTSI-G')])

# We can easily clean the data now:
ice2 = ice[ice.data_type != '-9999']
print('shape:', ice2.shape)
# And repeat the plot
x = ice2.year
y = ice2.extent
plt.scatter(x, y, color = 'red')
plt.xlabel('Month')
plt.ylabel('Extent')

plt.show()

sns.lmplot(x = "mo", y = "extent", data = ice2, height = 5.2, aspect = 2);
plt.savefig("D:\SSD_Python01012023\IceExtentCleanedByMonth.png", dpi = 300, bbox_inches = 'tight')
plt.show()

# Compute the mean for each month.
grouped = ice2.groupby('mo')
month_means = grouped.extent.mean()
month_variances = grouped.extent.var()
print('Means:', month_means)
print('Variances:',month_variances)

# Data normalization
for i in range(12):
    ice2.extent[ice2.mo == i+1] = 100*(ice2.extent[ice2.mo == i+1] - month_means[i+1])/month_means.mean()
    
sns.lmplot(x = "mo", y = "extent", data = ice2, height = 5.2, aspect = 2);
plt.savefig("D:\SSD_Python01012023\IceExtentNormalizedByMonth.png", dpi = 300, bbox_inches = 'tight')
plt.show()

print('mean:', ice2.extent.mean())
print('var:', ice2.extent.var())

sns.lmplot(x = "year", y = "extent", data = ice2, height = 5.2, aspect = 2);
plt.savefig("D:\SSD_Python01012023\IceExtentAllMonthsByYearlmplot.png", dpi = 300, bbox_inches = 'tight')
plt.show()


#For January
jan = ice2[ice2.mo == 1];
sns.lmplot(x = "year", y = "extent", data = jan, height = 5.2, aspect = 2);
plt.show()

# Calculates a Pearson correlation coefficient and the p-value for testing non-correlation.
import scipy.stats
scipy.stats.pearsonr(ice2.year.values, ice2.extent.values)

from sklearn.linear_model import LinearRegression

est = LinearRegression(fit_intercept = True)

x = ice2[['year']]
y = ice2[['extent']]

est.fit(x, y)

print("Coefficients:", est.coef_)
print("Intercept:", est.intercept_)

from sklearn import metrics

# Analysis for all months together.
x = ice2[['year']]
y = ice2[['extent']]
model = LinearRegression()
model.fit(x, y)
y_hat = model.predict(x)
plt.plot(x, y,'o', alpha = 0.5)
plt.plot(x, y_hat, 'r', alpha = 0.5)
plt.xlabel('year')
plt.ylabel('extent (All months)')
print("MSE:", metrics.mean_squared_error(y_hat, y))
print("R^2:", metrics.r2_score(y_hat, y))
print("var:", y.var())
plt.savefig("D:\SSD_Python01012023\IceExtentLinearRegressionAllMonthsByYearPrediction.png", dpi = 300, bbox_inches = 'tight')
plt.show()

# Analysis for a particular month.
x = jan[['year']]
y = jan[['extent']]

model = LinearRegression()
model.fit(x, y)

y_hat = model.predict(x)

plt.figure()
plt.plot(x, y,'-o', alpha = 0.5)
plt.plot(x, y_hat, 'r', alpha = 0.5)
plt.xlabel('year')
plt.ylabel('extent (January)')

print("MSE:", metrics.mean_squared_error(y_hat, y))
print("R^2:", metrics.r2_score(y_hat, y))

X = np.array([2025]).reshape(-1,1)
y_hat = model.predict(X)
j = 1 # January
# Original value (before normalization)
y_hat = (y_hat*month_means.mean()/100) + month_means[j]
print("Prediction of extent for January 2025 (in millions of square km):", y_hat)
