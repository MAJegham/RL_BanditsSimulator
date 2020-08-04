import numpy as np

def rollingAverage(values_p, window_p):
    average_l = []
    for i_cntr in range(len(values_p)):
        windowValues_l = values_p[max(0,i_cntr-window_p+1):i_cntr+1]
        average_l.append(np.mean(windowValues_l))
    return average_l

def progressBar(progress_p, total_p, barLength_p = 20):
    progressPercentage_l = float(progress_p) * 100 / total_p
    arrow_l   = '-' * int(progressPercentage_l/100 * barLength_p - 1) + '>'
    spaces_l  = ' ' * (barLength_p - len(arrow_l))

    print('Progress: [%s%s] %d %%' % (arrow_l, spaces_l, progressPercentage_l), end='\r')