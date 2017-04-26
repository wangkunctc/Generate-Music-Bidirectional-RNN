import numpy as np
import soundfile as sf
import os

def get_data_from_location(location):
    list_data = os.listdir(location)
    
    if len(list_data) == 0:
        print "not found any data, exiting.."
        exit(0)
    
    for i in list_data:
        if i.find('.wav') < 0 and i.find('.flac') < 0:
            print "Only support .WAV or .FLAC, exiting.."
            exit(0)
    
    samplerate = []
    
    for i in list_data:
        _, sample = sf.read(location + i)
        samplerate.append(sample)
    
    if len(set(samplerate)) > 1:
        print "make sure all sample rate are same, exiting.."
        exit(0)
        
    data, samplerate = sf.read(location + list_data[0])
    
    if len(data.shape) > 1:
        data = data[:, 0]
    
    for i in xrange(1, len(list_data), 1):
        
        sample_data, _ = sf.read(location + list_data[i])
        
        if len(sample_data.shape) > 1:
            sample_data = sample_data[:, 0]
        
        data = np.append(data, sample_data, axis = 0)
    
    return data, samplerate