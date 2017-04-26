import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", palette="muted")
import numpy as np

def generategraph(real_data, model, sess):
    
    fig = figure(figsize = (10,10))
    
    plt.subplot(2, 1, 1)
    plt.plot(real_data)
    plt.title("real music")