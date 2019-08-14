from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:47:53 2018

@author: Vladimir
"""

import sys
from SPA import SPA
from JTLA import JTLA
from JAMPA import JAMPA
from SPA_plotting import *


if __name__=='__main__':

    path = r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/SPA/DATA/'

    spa01 = SPA(path,'SPA01')
#    spa03 = SPA(path,'SPA03')
    spa04 = SPA(path,'SPA04')
#    spa05 = SPA(path,'SPA05')
    spa08 = SPA(path,'SPA08')
    spa13_v2 = SPA(path,'SPA13_v2')
    spa14_v2 = SPA(path,'SPA14_v2')
    spa04_v2 = SPA(path,'SPA04_v2')
#    spa14_v3 = SPA(path,'SPA14_v3')
#    spa14_v4 = SPA(path,'SPA14_v4')
#    spa32 = SPA(path,'SPA32')
#    spa33 = SPA(path,'SPA33')
#    spa34 = SPA(path,'SPA34')
    spa35 = SPA(path,'SPA35')
#    spa37_v2 = SPA(path,'SPA37_v2')
#    spa37 = SPA(path,'SPA37')
#    spa39 = SPA(path,'SPA39')
    spa34_v3 = SPA(path,'SPA34_v3')
#    spa34_v4 = SPA(path,'SPA34_v4')    
#    spa34_v5 = SPA(path,'SPA34_v5')
#    rsl003 = SPA(path,'RSL003')
#    spa34_v5 = SPA(path,'SPA34_v5')
    spa34_v6 = SPA(path,'SPA34_v6')
#    rsl04 = SPA(path,'RSL04')
#    spa34_v7 = SPA(path,'SPA34_v7')    
#    spa08_v2 = SPA(path,'SPA08_v2')  
    ppf02 = SPA(path,'PPFSPA02')

    
    path = r'/Users/vs362/Google Drive (vladimir.sivak@yale.edu)/Qulab/JAMPA/DATA/'   
    

    JAMPA01 = JAMPA(path,'JAMPA01',200)
#    JTLA02 = JTLA(path,'JTLA02',1000)
#    JTLA03 = JTLA(path,'JTLA03',1178)
#    JTLA06_v3 = JTLA(path,'JTLA06_v3',200) 
#    JTLA07_v1 = JTLA(path,'JTLA07_v1',1003)
    JTLA08 = JTLA(path,'JAMPA08',1003)
#    JTLA08_v2 = JTLA(path,'JTLA08_v2',1003)
    JTLA08_v3 = JAMPA(path,'JAMPA08_v3',1003)
    JAMPA007 = JAMPA(path,'JAMPA007',1003)
    JAMPA008 = JAMPA(path,'JAMPA008',1003)
    JAMPA09 = JAMPA(path,'JAMPA09',1000)
    JAMPA10 = JAMPA(path,'JAMPA10',1000)  
    JAMPA10_v2 = JAMPA(path,'JAMPA10_v2',1000)       
    