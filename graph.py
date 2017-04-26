import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", palette="muted")
import numpy as np

def generategraph(real_data, model, sess, samplerate, boundary, checkpoint = False):
    
    initial = np.random.randint(0, real_data.shape[0] / 2)
    fig = plt.figure(figsize = (10,15))
    
    if checkpoint:
        length = 10
    else:
        while True:
            length = raw_input("insert length of sound(sec): ")
            try:
                length = int(length)
                break
            except:
                print "enter INTEGER only"
    
    plt.subplot(3, 1, 1)
    plt.plot(real_data[initial : initial + (length * samplerate)])
    plt.title("real music")
    
    probs = model.step(sess, np.array([real_data[initial : initial + boundary]]), True)
    backward_generated = probs[0, :].tolist()
    forward_generated = probs[1, :].tolist()

    for x in xrange(0, length * (samplerate / boundary)):
        probs = model.step(sess, np.array([np.mean(probs, axis = 0)]), False)
        backward_generated += probs[0, :].tolist()
        forward_generated += probs[1, :].tolist()
        
    plt.subplot(3, 1, 2)
    plt.plot(backward_generated)
    plt.title("left generated")
    
    
    plt.subplot(3, 1, 3)
    plt.plot(forward_generated)
    plt.title("right generated")
    fig.tight_layout()
    plt.savefig('graph.png')
    plt.savefig('graph.pdf')

    
    