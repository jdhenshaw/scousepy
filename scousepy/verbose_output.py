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
                print('Number of spectra to fit manually: {}'.format(var))
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
        if step=='mid':
            print("")
            print('You fitted a total of {0} spectra in {1} minutes'.format(length, (t2-t1)/60.))
            print("")
            progress_bar=[]
        if step=='end':
            print("")
            print('scousepy stage 2 completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]

    if stage=='s3':
        if step=='start':
            print("")
            print("Beginning stage_3 analysis...")
            print("")
            progress_bar=[]

    return progress_bar
