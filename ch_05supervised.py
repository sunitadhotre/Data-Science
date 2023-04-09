import matplotlib.pylab as plt
import pickle

#%matplotlib inline 
plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=True)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('font', size=12) 
plt.rc('figure', figsize = (12, 5))


ofname = open ('D:\\SSD_Python01012023\\dataset_small.pkl','rb')
# x stores input data and y target values
(x,y) = pickle.load(ofname, encoding="bytes")

with open('D:\\SSD_Python01012023\\dataset_small.pkl', 'rb') as f:
    loaded = pickle.load(f, encoding="latin1")

    #pickle.load(open(model_file, 'rb'), encoding = 'bytes')
