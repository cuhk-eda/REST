# RSMT utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import ctypes
import heapq

    
class ArrayType:
    def __init__(self, type):
        self.type = type

    def from_param(self, param):
        typename = type(param).__name__
        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, ctypes.Array):
            return param
        else:
            raise TypeError('Can\'t convert % s' % typename)

    # Cast from lists / tuples
    def from_list(self, param):
        val = ((self.type)*len(param))(*param)
        return val

    from_tuple = from_list
    from_array = from_list
    from_ndarray = from_list

class Evaluator:
    def __init__(self):
        path = os.getcwd() + '/algorithms/libeval.so'
        eval_mod = ctypes.cdll.LoadLibrary(path)
        DoubleArray = ArrayType(ctypes.c_double)
        IntArray = ArrayType(ctypes.c_int)
        
        eval_mod.gst_open_geosteiner()
        
        eval_func = eval_mod.eval
        eval_func.argtypes = (DoubleArray, IntArray, ctypes.c_int)
        eval_func.restype = ctypes.c_double
        self.eval_func = eval_func
        
        gst_rsmt_func = eval_mod.call_gst_rsmt
        gst_rsmt_func.argtypes = (
            ctypes.c_int, DoubleArray, ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        gst_rsmt_func.restype = ctypes.c_int
        self.gst_rsmt_func = gst_rsmt_func

    def eval_batch(self, input_batch, output_batch, degree):
        lengths = []
        batch_size = len(input_batch)
        for i in range(batch_size):
            lengths.append(self.eval_func(input_batch[i].reshape(-1), output_batch[i], degree))
        return np.array(lengths)
        
    def gst_rsmt(self, inputs):
        degree = len(inputs)
        terms = inputs.flatten()
        length = ctypes.c_double()
        nsps = ctypes.c_int()
        sps = (ctypes.c_double * (degree * 2))()
        sps = ctypes.cast(sps, ctypes.POINTER(ctypes.c_double))
        nedges = ctypes.c_int()
        edges = (ctypes.c_int * (degree * 4))()
        ctypes.cast(edges, ctypes.POINTER(ctypes.c_int))
        self.gst_rsmt_func(degree, terms, ctypes.byref(length), 
            ctypes.byref(nsps), sps, ctypes.byref(nedges), edges)
        sp_list = []
        for i in range(nsps.value):
            sp_list.append([sps[2 * i], sps[2 * i + 1]])
        edge_list = []
        for i in range(nedges.value):
            edge_list.append([edges[2 * i], edges[2 * i + 1]])
        return length.value, sp_list, edge_list

edge_color = 'black'
edge_width = .5
term_color = 'black'
term_size = 4
steiner_color = 'black'

def plot_rest(input, output):
    x_low, y_low, x_high, y_high = [], [], [], []
    for i in range(len(input)):
        x_low.append(input[i][0])
        y_low.append(input[i][1])
        x_high.append(input[i][0])
        y_high.append(input[i][1])
    for i in range(int(len(output) / 2)):
        x_idx = output[2*i]
        y_idx = output[2*i+1]
        x = input[x_idx][0]
        y = input[y_idx][1]
        y_low[x_idx] = min(y_low[x_idx], y)
        y_high[x_idx] = max(y_high[x_idx], y)
        x_low[y_idx] = min(x_low[y_idx], x)
        x_high[y_idx] = max(x_high[y_idx], x)
    for i in range(len(x_low)):
        plt.plot([x_low[i], x_high[i]], [input[i][1], input[i][1]], '-', color=edge_color, linewidth=edge_width)
        plt.plot([input[i][0], input[i][0]], [y_low[i], y_high[i]], '-', color=edge_color, linewidth=edge_width)
    plt.plot(list(input[:,0]), list(input[:,1]), 's', color=term_color, markerfacecolor='black', markersize=term_size, markeredgewidth=edge_width)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    # fig.savefig('rsmt.pdf')

def plot_gst_rsmt(terms, sps, edges):
    degree = len(terms)
    points = np.concatenate([terms, sps], 0)
    for i in range(len(edges)):
        u = edges[i][0]
        v = edges[i][1]
        plt.plot([points[u][0], points[u][0]], [points[u][1], points[v][1]], '-', color=edge_color, linewidth=edge_width)
        plt.plot([points[u][0], points[v][0]], [points[v][1], points[v][1]], '-', color=edge_color, linewidth=edge_width)
    plt.plot([terms[i][0] for i in range(degree)], [terms[i][1] for i in range(degree)], 's', markerfacecolor='black', color=term_color, markersize=term_size, markeredgewidth=edge_width)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)        
    
def save_data(test_data, test_file):
    with open(test_file, 'w') as f:
        for data in test_data:
            f.write(' '.join(['{:.8f} {:.8f}'.format(term[0], term[1]) for term in data]))
            f.write('\n')
            
def read_data(test_file):
    with open(test_file, 'r') as f:
        test_data = []
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            data = [float(coord) for coord in line]
            data = np.array(data).reshape([-1, 2])
            test_data.append(data)
    return np.array(test_data)
        

def transform_inputs(inputs, t):
    # 0 <= t <= 7
    xs = inputs[:,:,0]
    ys = inputs[:,:,1]
    if t >= 4:
        temp = xs
        xs = ys
        ys = temp
    if t % 2 == 1:
        xs = 1 - xs
    if t % 4 >= 2:
        ys = 1 - ys
    return np.stack([xs, ys], -1)
