# Licensed under an MIT open source license - see LICENSE
import numpy as np
def generate_saa_list(scouseobject):
    """
    Returns a list constaining all spectral averaging areas.

    Parameters
    ----------
    scouseobject : Instance of the scousepy class

    """
    saa_list=[]
    for i in range(len(scouseobject.wsaa)):
        saa_dict = scouseobject.saa_dict[i]
        for key in saa_dict.keys():
            # get the relavent SAA
            saa = saa_dict[key]
            if saa.to_be_fit:
                saa_list.append([saa.index, i])

    return saa_list
