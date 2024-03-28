import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
import numpy as np
import math

def plot_vf(tds, vmin=None, vmax=None, show_value=True, show_colorbar=True, title=None, color='black'):
    vf_type = 24
    if len(tds) == 68:
        vf_type = 10
    mat = np.zeros([8, 9])
    nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,
             55,62,63,64,65,70,71] # 8x9, gray:[43,45]
    mat[3][7] = np.nan
    mat[4][7] = np.nan
    if vf_type == 10:
        mat = np.zeros([10, 10])
        nulls = [0,1,2,3,6,7,8,9,10,11,18,19,20,29,30,39,60,69,70,
                 79,80,81,88,89,90,91,92,93,96,97,98,99] # 10x10
    k = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            pos = i * mat.shape[1] + j
            if pos not in nulls:
                mat[i][j] = tds[k]
                k += 1
    if vmin == None:            
        vmin = math.floor(np.min(tds)*100)/100
    if vmax == None:            
        vmax = math.ceil(np.max(tds)*100)/100
    if show_value:
        fig, ax = plt.subplots(1, figsize=(6,5))
        aspect = 0.042
    else:
        fig, ax = plt.subplots(1, figsize=(4,4))
        aspect = 0.05
    cmap = plt.get_cmap('bwr_r').copy()
    cmap.set_bad('gray')
    if vmin >= 0:
        divnorm= colors.TwoSlopeNorm(vmin=-0.001, vcenter=0, vmax=vmax)
    elif vmax <= 0:
        divnorm= colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=0.001)
    else:
        divnorm= colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
    im = plt.imshow(mat, cmap=cmap, norm=divnorm, aspect='auto')
    ax.axis('off')
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        cbar = plt.colorbar(orientation="horizontal", cax=cax, ticks=[vmin, 0, vmax])
        cax.set_aspect(aspect)
        cbar.ax.tick_params(labelsize=12)
    if vf_type == 24:
        ax.add_patch(Rectangle((6.48, 2.48), 1, 2, fill=False, edgecolor='black', lw=0.5))
    if show_value:
        for (i, j), z in np.ndenumerate(mat):
            pos = i * mat.shape[1] + j
            if pos not in nulls:
                ax.text(j, i, round(z, 2), ha='center', va='center', size=11, color=color)
    if title != None:
        ax.set_title(title, size=12)
    plt.show()


def gen_vfmat(tds, mark_blind=False):
    vf_type = 24
    if len(tds) == 68:
        vf_type = 10
    mat = np.zeros([8, 9])
    nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,
             55,62,63,64,65,70,71] # 8x9, gray:[43,45]
    if vf_type == 10:
        mat = np.zeros([10, 10])
        nulls = [0,1,2,3,6,7,8,9,10,11,18,19,20,29,30,39,60,69,70,
                 79,80,81,88,89,90,91,92,93,96,97,98,99] # 10x10
    else:
        if mark_blind:
            mat[3][7] = np.nan
            mat[4][7] = np.nan
    k = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            pos = i * mat.shape[1] + j
            if pos not in nulls:
                mat[i][j] = tds[k]
                k += 1
    return mat

def plot_vf_mat(mat, vmin=None, vmax=None, show_value=True, show_colorbar=True):
    mat = np.array(mat)
    if vmin == None:            
        vmin = math.floor(np.min(mat)*100)/100 # np.min(mat)
    if vmax == None:            
        vmax = math.ceil(np.max(mat)*100)/100 # np.max(mat)
    vf_type = 24
    nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,
             55,62,63,64,65,70,71] # 8x9, gray:[43,45]
    if len(mat.flatten()) == 100:
        vf_type = 10
        nulls = [0,1,2,3,6,7,8,9,10,11,18,19,20,29,30,39,60,69,70,
                 79,80,81,88,89,90,91,92,93,96,97,98,99] # 10x10
    else:
        mat[3][7] = np.nan
        mat[4][7] = np.nan
    if show_value:
        fig, ax = plt.subplots(1, figsize=(6,5))
        aspect = 0.042
    else:
        fig, ax = plt.subplots(1, figsize=(4,4))
        aspect = 0.05
    cmap = plt.get_cmap('bwr_r').copy()
    cmap.set_bad('gray')
    if vmin >= 0:
        divnorm= colors.TwoSlopeNorm(vmin=-0.001, vcenter=0, vmax=vmax)
    elif vmax <= 0:
        divnorm= colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=0.001)
    else:
        divnorm= colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = plt.imshow(mat, cmap=cmap, norm=divnorm, aspect='auto')
    ax.axis('off')
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        cbar = plt.colorbar(orientation="horizontal", cax=cax, ticks=[vmin, 0, vmax])
        cax.set_aspect(aspect)
        cbar.ax.tick_params(labelsize=12)
        print(vmin, vmax)
    if vf_type == 24:
        ax.add_patch(Rectangle((6.48, 2.48), 1, 2, fill=False, edgecolor='black', lw=0.5))
    if show_value:
        for (i, j), z in np.ndenumerate(mat):
            pos = i * mat.shape[1] + j
            if pos not in nulls:
                ax.text(j, i, round(z, 2), ha='center', va='center', size=11)
    plt.show()
    

def cal_min(tds_combine):
    vmin = np.min(tds_combine[0])
    for i in range(1, len(tds_combine)):
        if vmin > np.min(tds_combine[i]):
            vmin = np.min(tds_combine[i])
    return vmin

def cal_max(tds_combine):
    vmax = np.max(tds_combine[0])
    for i in range(1, len(tds_combine)):
        if vmax < np.max(tds_combine[i]):
            vmax = np.max(tds_combine[i])
    return vmax
    
def visualize(imgs, sizes=(9,3), vmin=None, vmax=None, show_colorbar=False, show_value=False, title=None, aspect=0.042):
    
    if vmin == None:            
        vmin = math.floor(cal_min(imgs)*100)/100
    if vmax == None:            
        vmax = math.ceil(cal_max(imgs)*100)/100
    if vmin >= 0:
        divnorm= colors.TwoSlopeNorm(vmin=-0.001, vcenter=0, vmax=vmax)
    elif vmax <= 0:
        divnorm= colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=0.001)
    else:
        divnorm= colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
    fig = plt.figure(figsize=sizes, constrained_layout=True)
    for i, tds in enumerate(imgs):
        vf_type = 24
        mat = gen_vfmat(tds)
        nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,
             55,62,63,64,65,70,71] # 8x9, gray:[43,45]
        if len(tds) == 68:
            vf_type = 10
            nulls = [0,1,2,3,6,7,8,9,10,11,18,19,20,29,30,39,60,69,70,
                 79,80,81,88,89,90,91,92,93,96,97,98,99] # 10x10
        else:
            mat[3][7] = np.nan
            mat[4][7] = np.nan
        plt_idx = i+1
        ax = plt.subplot(1, len(imgs), plt_idx)
        ax.axis('off')
        
        cmap = plt.get_cmap('bwr_r').copy()
        cmap.set_bad('gray')
        im = ax.imshow(mat, cmap=cmap, norm=divnorm, aspect='auto')
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.1)
            cbar = fig.colorbar(im, orientation="horizontal", cax=cax, ticks=[vmin, 0, vmax])
            cax.set_aspect(aspect)
            cbar.ax.tick_params(labelsize=12)
        if vf_type == 24:
            ax.add_patch(Rectangle((6.5, 2.5), 1, 2, fill=False, edgecolor='black', lw=0.5))
        if show_value:
            for (p, q), z in np.ndenumerate(mat):
                pos = p * mat.shape[1] + q
                if pos not in nulls:
                    ax.text(q, p, round(z, 2), ha='center', va='center', size=11)
        if title[i] != None:
            ax.set_title(title[i], size=12)
        
    plt.show()
    
    
def central_points(vf_tds, n=12):
    vf_mat = gen_vfmat(vf_tds)
    points = []
    if n == 4:
        return np.array(vf_tds)[[22,23,30,31]]
    else:
        for i in range(2, 6):
            for j in range(3, 7):
                if n != 16:
                    if (i,j) in [(2,3), (2,6), (5,3), (5,6)]:
                        continue
                points.append(vf_mat[i][j])
    return points
	
def prepare_data(vfdata):
    categorical_feats = ['righteye']
    numerical_feats = ['age', 'duration', 'centralval', 'centralprob', 'md', 'mdprob', 'psd', 'timegap']

    test_dataset = {'age':[], 'righteye':[], 'duration':[], 'centralval':[], 'centralprob':[], 'md':[], 'mdprob':[],
                   'psd':[], 'timegap':[]}
    tds24_test = []
    for vf in vfdata:
        age, righteye, duration, centralval, centralprob, md, mdprob, psd, timegap, tds = vf
        test_dataset['age'].append(age)
        test_dataset['righteye'].append(float(righteye))
        test_dataset['duration'].append(duration)
        test_dataset['centralval'].append(centralval)
        test_dataset['centralprob'].append(centralprob)
        test_dataset['md'].append(md)
        test_dataset['mdprob'].append(mdprob)
        test_dataset['psd'].append(psd)
        test_dataset['timegap'].append(timegap)

        tds24_test.append(tds)

    vf24_pdf_test = pd.DataFrame.from_dict(test_dataset)
    vf24_pdf_test[categorical_feats] = vf24_pdf_test[categorical_feats].astype(str)
    vf24_pdf_test[numerical_feats] = vf24_pdf_test[numerical_feats].astype(float)
    test_dataset = {}
    for key, value in vf24_pdf_test.items():
        test_dataset[key] = value[:, tf.newaxis]

    tds24_test = np.array(tds24_test, dtype=int)
    
    return test_dataset, tds24_test