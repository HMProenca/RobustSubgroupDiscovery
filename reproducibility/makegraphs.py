# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:34:22 2020

@author: Hugo Manuel Proenca
"""

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.markers as marker
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def make_graph(results,x_str,y_str,size_marker,color,typeofplot,separate_colour = 2):
    alp = 1
    fig= plt.figure()
    sep = separate_colour
    for iname,name in enumerate(results):  
        x = results[name][x_str]
        y = results[name][y_str]
        if typeofplot == "semilogy":
            plt.semilogy(x, y,alpha =alp,c=np.array(color[sep*iname])/255, marker='o',label=name,\
                linewidth = 0.5,markersize = size_marker)
        elif typeofplot == "plot":
            plt.plot(x, y,alpha =alp,c=np.array(color[sep*iname])/255, marker='o',label=name,\
                linewidth = 0.5,markersize = size_marker)
        elif typeofplot == "semilogx":
            plt.semilogx(x, y,alpha =alp,c=np.array(color[sep*iname])/255, marker='o',label=name,\
                linewidth = 0.5,markersize = size_marker)   
        elif typeofplot == "loglog":
            plt.loglog(x, y,alpha =alp,c=np.array(color[sep*iname])/255, marker='o',label=name,\
                linewidth = 0.5,markersize = size_marker) 
    plt.grid(b=True, which='major', axis='y', linestyle= '--', linewidth=0.6)
    #plt.ticklabel_format(style='plain')   
    plt.grid(b=True, which='major', axis='y', linestyle= '--', linewidth=0.6)
    lgd =plt.legend(loc='upper right')
    return fig,lgd