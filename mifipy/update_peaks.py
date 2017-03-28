import numpy as np
import pandas as pd
from scipy import optimize

def nearest_index_of(arrval, value):
    return np.argmin( abs(arrval - value) )

def gaussian_p(x, p):
    amp, mean, sigma = p
    return - amp / sigma**3 / np.sqrt(2 * np.pi) * (x - mean) * \
        np.exp(- (x - mean)**2. / 2 / sigma**2.)

def gaussian_pp(x, p):
    amp, mean, sigma = p
    return amp / sigma**5 / np.sqrt(2 * np.pi) * (x - mean) ** 2. * np.exp(- (x - mean)**2. / 2 / sigma**2.) \
        - amp / sigma**3 / np.sqrt(2 * np.pi) * np.exp(- (x - mean)**2. / 2 / sigma**2.) 

# An Ugly Function to Find 5 peaks beased on 1st derivative
def find_five_gaussians_roots(parameters):
    try:
        p1, p2, p3, p4, p5 = parameters.reshape(5, 3)
    except:
        p1, p2, p3, p4, p5 = np.array(parameters).reshape(5, 3)

    def five_gaussian_p(x):
        return sum([gaussian_p(x, p) for p in [p1, p2, p3, p4, p5]])
    
    def five_gaussian_pp(x):
        return sum([gaussian_pp(x, p) for p in [p1, p2, p3, p4, p5]])
    
    center_list = [p1[1], p2[1], p3[1], p4[1], p5[1]]
    opt_list = np.array(sorted([optimize.root(five_gaussian_p, i, jac=five_gaussian_pp).x[0] 
                                for i in center_list]))
    return [opt_list[nearest_index_of(opt_list, p[1])] for p in [p1, p2, p3, p4, p5]]

# update the values of peaks to replace gaussian centers
def updating_dataframe_peaks_with_roots(df):
    df_update = df.copy()
    peaks_string_list = ['peak1', 'peak2', 'peak3', 'peak4', 'peak5']
    
    # Best fit with 5 gaussians
    centers = pd.DataFrame([df_update.loc[:, pkstr] for pkstr in peaks_string_list]).T
    centers.columns = ['center1', 'center2', 'center3', 'center4', 'center5']

    for i in df_update.index:
        parameters = [df_update.loc[i, par + str(j)] for j in range(1,6) for par in ['amp', 'peak', 'sigma']]
        five_roots = find_five_gaussians_roots(parameters)
        for root, pkstr in zip(five_roots, peaks_string_list): 
            df_update.loc[i, pkstr] = root

    # concat df_update and centers
    return pd.concat([df_update, centers], axis=1)
    
# gaussian_fit_helper updater
def update_gaussuan_helper_with_roots(result):
    parameters = [result.best_values[prefix + par] 
        for prefix in ['g1_', 'g2_', 'g3_', 'g4_', 'g5_']
        for par in ['amplitude', 'center', 'sigma']]
    peak_list = [prefix + 'peak' for prefix in ['g1_', 'g2_', 'g3_', 'g4_', 'g5_']]
    peaks = find_five_gaussians_roots(parameters)
    for name, peak in zip(peak_list, peaks):
        result.best_values[name] = peak
    return result

def updating_dictionary_peaks_with_roots(df):
    df_update = df.copy()
    peaks_string_list = ['g1_peak', 'g2_peak', 'g3_peak', 'g4_peak', 'g5_peak']
    parameters_list = ['g' + str(j) + '_' + par for j in range(1,6) 
                                                for par in ['amplitude', 'center', 'sigma']]
    peaks = {p : [] for p in peaks_string_list}
    for i in df_update.index:
        parameters = [df_update.loc[i, par] for par in parameters_list]
        five_roots = find_five_gaussians_roots(parameters)

        for root, pkstr in zip(five_roots, peaks_string_list): 
            peaks[pkstr].append(root)

    # concat df_update and centers
    return pd.concat([pd.DataFrame(peaks), df_update], axis=1)
