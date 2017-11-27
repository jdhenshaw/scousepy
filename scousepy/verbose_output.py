import numpy as np
from .progressbar import AnimatedProgressBar

def print_to_terminal(stage='', step='', length=None, var=None, t1=None, t2=None):
    """
    Keeping all the noisy stuff in one place.
    """
    if stage=='s1':
        if step=='start':
            print('')
            print('----------------')
            print('scousepy fitting')
            print('----------------')
            print('')
            print('Beginning stage_1 analysis...')
            print('')
            progress_bar=[]

        if step=='moments':
            print('Calculating moments...')
            print('')
            progress_bar=[]

        if step=='coverage':
            if length != None:
                print('Establishing coverage...')
                progress_bar = AnimatedProgressBar(end=length-1, width=50, \
                                                   fill='=', blank='.')
                print('')
            else:
                print('Number of spectra to fit manually: {}'.format(np.count_nonzero(~np.isnan(np.asarray(var)[:,0]))))
                progress_bar=[]
                print('')

        if step=='end':
            print('Total number of spectra: {}'.format(length))
            print('scousepy stage 1 completed in: {} minutes'.format((t2-t1)/60.))
            print('')
            progress_bar=[]

    if stage=='s2':
        if step=='start':
            print('')
            print('Beginning stage_2 analysis...')
            print('')
            progress_bar=[]

    return progress_bar
