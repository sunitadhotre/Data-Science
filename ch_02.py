import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib parameters
plt.rc('text', usetex=True)
plt.rc('font', family='times', size=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Create data
data = {
    'year': [2010, 2011, 2012, 2010, 2011, 2012, 2010, 2011, 2012],
    'team': ['FCBarcelona', 'FCBarcelona', 'FCBarcelona', 'RMadrid', 'RMadrid', 'RMadrid', 'ValenciaCF', 'ValenciaCF', 'ValenciaCF'],
    'wins': [30, 28, 32, 29, 32, 26, 21, 17, 19],
    'draws': [6, 7, 4, 5, 4, 7, 8, 10, 8],
    'losses': [2, 3, 2, 4, 2, 5, 9, 11, 11]
}

# Create a DataFrame
football = pd.DataFrame(data, columns=['year', 'team', 'wins', 'draws', 'losses'])

# Display the DataFrame
print(football)
