# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

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
        for j in range(len(saa_dict.keys())):
            # get the relavent SAA
            SAA = saa_dict[j]
            if SAA.to_be_fit:
                saa_list.append([SAA.index, i])

    return saa_list
