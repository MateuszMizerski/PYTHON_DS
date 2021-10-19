#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import math
import numpy as np
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models.widgets import Select
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.layouts import column
import pandas as pd
from bokeh.models import LinearAxis

import time



def iteracja_energy(e):
    dx = 0.025                       #krok siatki w j.a 
    N = int(20/dx)+1
    Fi = np.zeros([N])               #tablica funkcji falowej
    V = np.zeros([N])                #tab poencjalu
    #definicja potencjalu
    #*0,07357 -def bezwymiarowej
    for i in range (1,int(0.25/dx)):
        V[i] = 300*0.07537
        V[N-1-i] = 300*0.07537
    
    V[0] = 100
    V[N-1] = 100
    
    for i in range (1,9):
        V[int((0.25+2.0*i)/dx)] = 300*0.07537/2.0
        V[int((1.75+2.0*i)/dx)] = 300*0.07537/2.0
        
    for i in range (0,9):
        for j in range (1,int(0.5/dx)):
            V[int((1.75+2.0*i)/dx)+j] = 300*0.07537
    
    X, F = [],[]
        
    energy = (x for x in np.arange(1.0, 1000.0, 0.005))
    Fi_N,E_R = [],[]
    Fi[0] = 0.0
    Fi[1] = 1.0
    
    X.append(0.0)
    X.append(dx)
    F.append(Fi[0])
    F.append(Fi[1])
    
    E = e *0.07537
    xno = 0
    for i in range(1,N-1):
        Fi[i+1] = Fi[i]*(2.0+dx*dx*(V[i]-E))-Fi[i-1]
        X.append((i+1)*dx)
        F.append(Fi[i+1])
    
    return X,F

def iteracja_energy2(e):
    dx = 0.025                       #krok siatki w j.a 
    N = int(20/dx)+1
    Fi = np.zeros([N])               #tablica funkcji falowej
    V = np.zeros([N])                #tab poencjalu
    #definicja potencjalu
    
    X, F = [],[]
        
    energy = (x for x in np.arange(1.0, 1000.0, 0.005))
    Fi_N,E_R = [],[]
    Fi[0] = 0.0
    Fi[1] = 1.0
    
    X.append(0.0)
    X.append(dx)
    F.append(Fi[0])
    F.append(Fi[1])
    
    E = e *0.07537
    xno = 0
    for i in range(1,N-1):
        Fi[i+1] = Fi[i]*(2.0+dx*dx*(V[i]-E))-Fi[i-1]
        X.append((i+1)*dx)
        F.append(Fi[i+1])
    
    return X,F


ER,ER1 = [],[]
file  = open("1.dat","r+")
for line in file:
    f_line = line.split()
    ER.append(float(f_line[0]))

file1  = open("2.dat","r+")
for line in file1:
    f_line = line.split()
    ER1.append(float(f_line[0]))


# In[2]:


def okno1(*args):
    df = pd.DataFrame()
    t1, t2 = iteracja_energy(ER[0])
    df['x'] = t1
    df['y'] = t2
    for e in ER:
        x,f = iteracja_energy(e)
        df[str(e)] = f
    source = ColumnDataSource(df)
    strs = [str(e) for e in ER]

    output_file('energie.html', title='Funkcja falowa')
    fig = figure(title='Funkcja falowa',
                 plot_height=600, plot_width=600,
                 toolbar_location='right')
    select = Select(title="Energia stanu własnego:", value=str(ER[0]), options=strs)

    fig.xaxis.axis_label = ("x")
    #fig.yaxis.axis_label = '\Psi'

    fig.line('x', 'y', source=source,
             color='gray', line_width=1)

    code = """
            var data = source.data;
            data['y'] = data[cb_obj.value];

            source.change.emit();
    """
    callback = CustomJS(args=dict(source=source), code=code)
    select.callback = callback
    layout = column(select, fig)
    show(layout)
    


# In[3]:


def okno2(*args):
    df = pd.DataFrame()
    t1, t2 = iteracja_energy(ER1[0])
    df['x'] = t1
    df['y'] = t2
    for e in ER1:
        x,f = iteracja_energy2(e)
        df[str(e)] = f
    source = ColumnDataSource(df)
    strs = [str(e) for e in ER1]

    output_file('energie2.html', title='Funkcja falowa')
    fig = figure(title='Wykresik',
                 plot_height=600, plot_width=600,
                 toolbar_location='right')
    select = Select(title="Energia stanu własnego", value=str(ER1[0]), options=strs)
    fig.line('x', 'y', source=source,
             color='gray', line_width=1)

    code = """
            var data = source.data;
            data['y'] = data[cb_obj.value];

            source.change.emit();
    """
    callback = CustomJS(args=dict(source=source), code=code)
    select.callback = callback
    layout = column(select, fig)
    show(layout)


# In[4]:


okno1()
okno2()


# In[ ]:




