import math
import numpy as np
import matplotlib.pyplot as plt

from src.visualisation import get_direction_from_trig
from src.geometry import sq_distance


def plot_error_graph(data, label, x="weighted bi-directional chamfer loss", y="No. of elements", max_val=None):
    # filter top 99% to remove outliers
    sorted_data = np.sort(data)[::-1]
    cap = int(len(data)/100)
    filtered_data = sorted_data[cap:]
    norm = np.linalg.norm(filtered_data, ord=1)
    norm_data = filtered_data/norm
    print("avg", np.average(filtered_data), cap, filtered_data[0], filtered_data[-1])
    
    # draw graph
    fig = plt.figure(figsize=(12,4))
    steps = 250
    if max_val is None:
        n, bins, _ = plt.hist(filtered_data, bins=np.arange(0,filtered_data[0],(filtered_data[0] - 0)/steps))
    else:
        n, bins, _ = plt.hist(filtered_data, bins=np.arange(0,max_val,(max_val - 0)/steps))
    mid = 0.5*(bins[1:] + bins[:-1])
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(label)
    plt.errorbar(mid, n, yerr=0.01, fmt='none')
    
    
# plot percentage error of predicted parameters
# direction errors are calculated as angle deviations (in degrees)
# dimension / angle errors are calculated as a percentage of target label
# position errors are plotted as a percentage of target radius
def plot_single_parameter_error(labels_list, preds_list, k, param_type, label, radius_index=0, max_val=None):
    errors = []
    error_count = 0
    for i, pr in enumerate(preds_list):
        if param_type == 'direction':
            pred = get_direction_from_trig(pr, k) 
            y = get_direction_from_trig(labels_list[i], k)
            try:
                errors.append(math.degrees(math.acos(np.dot(pred, y))))
            except:
                error_count += 1
            
        elif param_type == 'dimension':
            deviation = abs(pr[k] -  labels_list[i][k])
            errors.append(deviation / labels_list[i][k])
                                        
        elif param_type == 'angle':
            pred = math.degrees(math.atan2(pr[k], pr[k+1]))
            target = math.degrees(math.atan2(labels_list[i][k], labels_list[i][k+1]))
            deviation = abs(pred - target)
            errors.append(deviation / target)
                            
        elif param_type == 'position':
            pred = [pr[k], pr[k+1], pr[k+2]]
            y = [labels_list[i][k], labels_list[i][k+1], labels_list[i][k+2]]
            deviation = sq_distance(pred[0], pred[1], pred[2], 
                                      y[0], y[1], y[2])
            errors.append(deviation / labels_list[i][radius_index])
    
    if error_count > 0:
        print("errors", error_count)
    plot_error_graph(errors, "Parameter deviation", x=label, max_val=max_val)
    

def plot_parameter_errors(labels_list, preds_list, cat):
    if cat == 'pipe':
        plot_single_parameter_error(labels_list, preds_list, 0, 'dimension', 'radius')
        plot_single_parameter_error(labels_list, preds_list, 1, 'dimension', 'length')
        plot_single_parameter_error(labels_list, preds_list, 2, 'position', 'position', radius_index=0)
        plot_single_parameter_error(labels_list, preds_list, 5, 'direction', 'axis')
        
    elif cat == 'elbow':
        plot_single_parameter_error(labels_list, preds_list, 0, 'dimension', 'radius')
        plot_single_parameter_error(labels_list, preds_list, 1, 'dimension', 'x')
        plot_single_parameter_error(labels_list, preds_list, 2, 'dimension', 'y')
        plot_single_parameter_error(labels_list, preds_list, 6, 'angle', 'angle')
        plot_single_parameter_error(labels_list, preds_list, 3, 'position', 'position', radius_index=0)
        plot_single_parameter_error(labels_list, preds_list, 8, 'direction', 'axis')
        
    elif cat == 'tee':
        plot_single_parameter_error(labels_list, preds_list, 0, 'dimension', 'radius1')
        plot_single_parameter_error(labels_list, preds_list, 1, 'dimension', 'length1')
        plot_single_parameter_error(labels_list, preds_list, 2, 'dimension', 'radius2')
        plot_single_parameter_error(labels_list, preds_list, 3, 'dimension', 'length2')
        plot_single_parameter_error(labels_list, preds_list, 4, 'position', 'position', radius_index=0)
        plot_single_parameter_error(labels_list, preds_list, 7, 'direction', 'axis1')
        plot_single_parameter_error(labels_list, preds_list, 13, 'direction', 'axis2')
               