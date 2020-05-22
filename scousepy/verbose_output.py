# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
from astropy.utils.console import ProgressBar
from .colors import *
from tqdm import tqdm

def print_to_terminal(stage='', step='', length=None, var=None, t1=None, t2=None):
    """
    Keeping all the noisy stuff in one place.
    """
    if stage=='init':
        if step=='init':
            print('')
            print('---------------------')
            print(colors.fg._lightblue_+'initialising scousepy'+colors._endc_)
            print('---------------------')
            print('')
            progress_bar=[]
        if step=='configexists':
            print(colors.fg._lightgreen_+"scousepy config file already exists. Returning filepath. "+colors._endc_)
            print('')
            progress_bar=[]
        if step=='makingconfig':
            print(colors.fg._lightgreen_+"config file created "+colors._endc_)
            print('')
            progress_bar=[]
    if stage=='s1':
        if step=='load':
            print('')
            print('--------')
            print(colors.fg._lightblue_+'scousepy'+colors._endc_)
            print('--------')
            print('')
            print(colors.fg._lightblue_+"loading s1.scousepy...  "+colors._endc_)
            print('')
            progress_bar=[]
        if step=='start':
            print('')
            print('--------')
            print(colors.fg._lightblue_+'scousepy'+colors._endc_)
            print('--------')
            print('')
            print(colors.fg._lightblue_+'Beginning stage_1 analysis...'+colors._endc_)
            print('')
            progress_bar=[]
        if step=='moments':
            print('Calculating moments...')
            print('')
            progress_bar=[]
        if step=='coverage':
            if length != None:
                print('Generating SAAs of size: {}'.format(var))
                print("")
                progress_bar = tqdm(total=length)
            else:
                print("")
                print('Number of spectra to fit manually: {}'.format(var))
                progress_bar=[]
                print('')
        if step=='end':
            print('')
            print('Total number of spectra: {}'.format(length))
            print('scousepy stage 1 completed in: {} minutes'.format((t2-t1)/60.))
            print('')
            progress_bar=[]

    if stage=='s2':
        if step=='load':
            print('')
            print(colors.fg._lightblue_+"loading s2.scousepy...  "+colors._endc_)
            print('')
            progress_bar=[]
        if step=='start':
            print('')
            print(colors.fg._lightblue_+'Beginning stage_2 analysis...'+colors._endc_)
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
        if step=='load':
            print('')
            print(colors.fg._lightblue_+"loading s3.scousepy...  "+colors._endc_)
            print('')
            progress_bar=[]
        if step=='start':
            print("")
            print(colors.fg._lightblue_+"Beginning stage_3 analysis..."+colors._endc_)
            progress_bar=[]
        if step=='init':
            if length != None:
                print("")
                print('Initialising spectra...')
                print("")
                progress_bar = tqdm(total=length, position=0, leave=True)
        if step=='initend':
            print("")
            print('Intialisation completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]
        if step=='fitinit':
            print("")
            print('Fitting spectra...')
            print("")
            progress_bar = []
        if step=='fitting':
            if length != None:
                progress_bar = length
        if step=='fitend':
            print("")
            print('Fitting completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]
        if step=='compileinit':
            print("")
            print("Compiling solutions...")
            print("")
            progress_bar=[]
        if step=='compileend':
            print("")
            print('Compilation completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]


        if step=='merge':
            print("")
            print("Merging model solutions...")
            print("")
            progress_bar=[]
        if step=='duplicates':
            print("")
            print("Removing duplicate model solutions...")
            print("")
            progress_bar=[]
        if step=='end':
            print("")
            print('scousepy stage 3 completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]

    if stage=='s4':
        if step=='start':
            print("")
            print(colors.fg._lightblue_+"Beginning stage_4 analysis..."+colors._endc_)
            print("")
            progress_bar=[]
        if step=='end':
            print("")
            print('scousepy stage 4 completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]

    if stage=='s5':
        if step=='start':
            print("")
            print(colors.fg._lightblue_+"Beginning stage_5 analysis..."+colors._endc_)
            print("")
            progress_bar=[]
        if step=='end':
            print("")
            if var == 1:
                print("A single spectrum has been chosen for inspection.")
            else:
                print("A total of {0} spectra have been chosen for inspection\nThis includes {1} block(s) and {2} individual pixel(s).".format(var[0], var[1], var[2]))
            print("")
            print('scousepy stage 5 completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]

    if stage=='s6':
        if step=='start':
            print("")
            print(colors.fg._lightblue_+"Beginning stage_6 analysis..."+colors._endc_)
            print("")
            progress_bar=[]
        if step=='fitting':
            if length != None:
                print("")
                print('Automated fitting: wsaa = {0}'.format(var))
                print("")
                progress_bar = tqdm(total=length)
        if step=='end':
            print("")
            print('scousepy stage 6 completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]
    return progress_bar
