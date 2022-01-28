# Licensed under an MIT open source license - see LICENSE

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
            print('Generating SAAs of size: {}'.format(var))
            print("")
            progress_bar = []
        if step=='coverageend':
            print('')
            print('Updating SAA dictionary'.format(var))
            print("")
            progress_bar = tqdm(total=length, position=0, leave=True)
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
        if step=='modelselectstart':
            print("")
            print('Selecting best-fitting solutions...')
            print("")
            progress_bar=[]
        if step=='selectmodsstart':
            if length != None:
                progress_bar = tqdm(total=length, position=0, leave=True)
        if step=='modelselectend':
            print("")
            print('Model selection completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]
        if step=='end':
            print("")
            print('scousepy stage 3 completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]

    if stage=='s4':
        if step=='load':
            print('')
            print(colors.fg._lightblue_+"loading s4.scousepy...  "+colors._endc_)
            print('')
            progress_bar=[]
        if step=='start':
            print("")
            print(colors.fg._lightblue_+"Beginning stage_4 analysis..."+colors._endc_)
            print("")
            progress_bar=[]
        if step=='diagnosticsinit':
            print('Creating diagnostic maps...')
            print("")
            progress_bar=[]
        if step=='diagnosticsload':
            print('Diagnostic maps already created. Loading...')
            print("")
            progress_bar=[]
        if step=='diagnostics':
            if length != None:
                progress_bar = tqdm(total=length, position=0, leave=True)
        if step=='end':
            if np.size(var) == 1:
                print("A single spectrum was inspected.")
            else:
                print("A total of {0} spectra were inspected".format(np.size(var)))
            print("")
            print('scousepy stage 4 completed in: {0} minutes'.format((t2-t1)/60.))
            print("")
            progress_bar=[]

    return progress_bar
