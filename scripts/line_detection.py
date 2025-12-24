#!/usr/bin/env python3

# find the emission line(s) in each spec txt
# find the line first --> define line-free window --> subtract cont --> re-find the line and re-calc the snr 
# the output results: spec_txt_name (without the .result.txt) + _snmax_output.txt

import numpy as np 
import re, glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as ticker
import sys
import argparse
from scipy.optimize import curve_fit


def calc_other_line_snr(freq,flux,eflux,tar_freq,tar_vel_width):    
    freq_width = tar_vel_width/2 * tar_freq/299792.458
    chan_id_low = np.argmin(np.abs(freq-(tar_freq-freq_width)))
    chan_id_high = np.argmin(np.abs(freq-(tar_freq+freq_width))) 
    sn_line = np.sum(flux[chan_id_low:chan_id_high+1]/eflux[chan_id_low:chan_id_high+1]**2)/np.sqrt(np.sum(1/eflux[chan_id_low:chan_id_high+1]**2))
    return chan_id_low,chan_id_high,sn_line

def line_cont(x, a, b):
    b=0 # change to the 0-order line fit
    return a + b * x
def contsub(freq,flux):
    x = freq
    y = flux
    initial_guess = [0,0] 
    params, _ = curve_fit(line_cont, x, y, p0=initial_guess)
    a_fit, b_fit = params
    x_fit = np.linspace(x.min(), x.max(), len(x))
    y_fit = line_cont(x_fit, a_fit, b_fit)
    print('continuum:'+str(b_fit)+'*x+'+str(a_fit))
    return a_fit, b_fit 

def find_highest_snr_line(ch,min_nchan,max_nchan,f,flux,sn,eflux):
    for l in range(0,ch-min_nchan):   
        start = min_nchan-1
        if l+max_nchan-1>ch:
            end = ch+1
        else:
            end = l+max_nchan
        f[l,l+start:end] = np.cumsum(flux[l:end])[start:]
        sn[l,l+start:end] = np.cumsum(flux[l:end]/eflux[l:end]**2)[start:]/np.sqrt(np.cumsum(1/eflux[l:end]**2))[start:]
    snmax = np.nanmax(sn)
    vi, vj = np.unravel_index(np.nanargmax(sn), sn.shape)  # get lower & upper channels
    
    return snmax,vi,vj



def line_detect(path,src,file,vel_width_low=200,vel_width_high=500,snr_other_lines=False,line_info_file='',guess_redshift=0,break_set=5):
    #################################################
    ### need to write a line candidates csv file  ###
    ### the col_names: line_name, rest_freq       ###
    ### the content: e.g., "CO21,230.538"         ###
    #################################################
    
    os.chdir(path+'/'+src+'/uv_spec/')
    
    # combined_data = np.hstack([np.loadtxt(file,comments='#',usecols=(0,1,2), unpack=True) for file in files])
    line_set = []

    base = os.path.basename(file)
    parts = base.split('.')
    if len(parts) > 2:
        out_base = '.'.join(parts[:-2])
    else:
        out_base = parts[0]
   
    data = np.loadtxt(file,comments='#',usecols=(0,1,2), unpack=True)
    df = pd.DataFrame(data).T
    tmp = df.sort_values(by=df.columns[0])
    tmp = tmp[(tmp[2] != -99)&(tmp[1] != 0)].reset_index(drop=True)
    # tmp = tmp[(tmp[2] != -99)].reset_index(drop=True)

    ## produce seperated freq list --> find line in each interval
    freq_list = tmp[0]
    interval_ind_list = []
    interval_start = 0
    

    for i in range(len(freq_list)-1):
        if freq_list[i+1]-freq_list[i] > break_set:
            interval_end = i
            interval_ind_list.append([interval_start,interval_end+1])
            interval_start = i+1        
    interval_ind_list.append([interval_start,len(freq_list)])   

    for start,end in interval_ind_list:
        tmp_sub = tmp.iloc[start:end]
        ch = tmp_sub.shape[0]
        f = np.ndarray((ch,ch))
        sn = np.ndarray((ch,ch))
        f[:]=-1000
        sn[:]=-1000

        num = 1
        freq, flux, eflux = tmp_sub.values.T    
        vel = np.array([(np.median(freq)-freq[i])/np.median(freq)*299792.458 for i in range(len(freq))])
        velwidth = (np.diff(freq) / np.median(freq)) * 299792.458
        velwidth_med = np.nanmedian(velwidth)
        eflux[eflux==0] = np.median(eflux[eflux!=0])
        min_nchan = int(vel_width_low/velwidth_med/2)*2
        max_nchan = int(vel_width_high/velwidth_med/2+1)*2

        ## first find the line from the spec with continuum
        snmax,vi,vj = find_highest_snr_line(ch,min_nchan,max_nchan,f,flux,sn,eflux)
        
        ## subtract the cont through fitting on line-free window
        flux_noline_low = flux[:int(vi-min_nchan/2)]
        flux_noline_high = flux[int(vj+min_nchan/2):ch]
        freq_noline_low = freq[:int(vi-min_nchan/2)]
        freq_noline_high = freq[int(vj+min_nchan/2):ch]
        flux_noline = np.hstack((flux_noline_low,flux_noline_high))
        freq_noline = np.hstack((freq_noline_low,freq_noline_high))
        b,k = contsub(freq_noline,flux_noline)
        flux = flux-(b+k*freq)  ## contsub flux
        
        ## re-calc / re-find the highest-snr line on contsub-spec
        snmax,vi,vj = find_highest_snr_line(ch,min_nchan,max_nchan,f,flux,sn,eflux)
        
        velwidth_for_line = (freq[vi+1]-freq[vi])/freq[vi]*299792.458
        flux_int = np.nansum(flux[vi:vj])*velwidth_for_line
        signal_int = np.nansum(flux[vi:vj])
        vlo = vel[vi]
        vhi = vel[vj]
        freq_lo = freq[vi]
        freq_hi = freq[vj]
        line_width = vlo - vhi
        freq_middle = (freq_lo + freq_hi)/2

        snr_ch_lower = (flux/eflux)[:int(vi-min_nchan/2)]
        snr_ch_upper = (flux/eflux)[int(vj+min_nchan/2):ch]
        snr_ch = np.hstack((snr_ch_lower,snr_ch_upper))
        print('### The noise level of this spectrum: ',np.std(flux_noline))
        print('### The snr level of this spectrum: ',np.std(snr_ch))
        snr_ch_sigma = np.std(snr_ch)
        snr_cor = snmax/snr_ch_sigma # correct for the rms noise of the continuum
        snr_fin = signal_int/np.std(flux_noline)/np.sqrt(vj-vi)
        ## calculate the chance of detecting a spurious line
        p0 = (norm.cdf(snr_cor,0,1) - norm.cdf(-1.*snr_cor,0,1))
        trials = 10 * (ch/np.abs(vi-vj) * np.abs(vi-vj)**0.58 * np.log10(max_nchan/min_nchan)) 
        prob_spurious = 1 - p0**trials  # Jin+2019, Eq.1

        line = (num, snr_fin, snr_cor, vi+1, vj+1, freq_lo, freq_hi, vlo, vhi, line_width, flux_int, prob_spurious)
        line_set.append(line)  

    ## start to calculate the snr of other lines with consistent z
        if snr_other_lines:
            other_lines_info = []
            line_info = pd.read_csv(line_info_file)
            obs_freq = line_info['rest_freq']/(1+guess_redshift)
            flag_same_line = False
            for i in range(len(obs_freq)):

                if np.abs(freq_middle-obs_freq[i])/obs_freq[i]*3*10**5<300:
                    tar_freq = freq_middle
                    guess_redshift_ = line_info['rest_freq'][i]/tar_freq-1
                    flag_same_line = True
                    print('###### The code successfully finds one supposed line, now use the updated z = ',guess_redshift_)
                    break
            if flag_same_line:
                obs_freq = line_info['rest_freq']/(1+guess_redshift_)
                line_width_for_others = line_width
            else:
                print('############ The code does not find the supposed line! ###############')
                candidate_widths = list(range(vel_width_low, vel_width_high+1, 100))


            for i in range(len(obs_freq)):
                tar_freq = obs_freq[i]
                if tar_freq>freq[0] and tar_freq<freq[-1]:
                    if flag_same_line:
                        chan_id_low,chan_id_high,sn_line = calc_other_line_snr(freq,flux,eflux,tar_freq,line_width_for_others)
                        sn_cor_line = sn_line/snr_ch_sigma
                        other_lines_info.append([chan_id_low,chan_id_high,sn_cor_line])

                    ## if the code could not find the suppposed line, we calc the snr for width ranging 
                    ## from the vel_width_low to the vel_width_high with interval=100, and then select the best one to record
                    else:
                        snr_with_diff_width = []
                        line_info_with_diff_width = []
                        for w in candidate_widths:
                            chan_id_low_,chan_id_high_,sn_line_ = calc_other_line_snr(freq,flux,eflux,tar_freq,w)
                            line_info_with_diff_width.append((chan_id_low_,chan_id_high_,sn_line_,w))
                            snr_with_diff_width.append(sn_line_/snr_ch_sigma)
                        width_id = np.argmax(snr_with_diff_width)
                        sn_cor_line = snr_with_diff_width[width_id]
                        
                        chan_id_low,chan_id_high,sn_line,line_width_for_others = line_info_with_diff_width[width_id]
                        other_lines_info.append([chan_id_low,chan_id_high,sn_cor_line])
                    
                    velwidth_for_line_ = (freq[chan_id_low+1]-freq[chan_id_low])/freq[chan_id_low]*299792.458
                    flux_int_line = np.nansum(flux[chan_id_low:chan_id_high])*velwidth_for_line_
                    sn_fin_line = np.nansum(flux[chan_id_low:chan_id_high])/np.std(flux_noline)/np.sqrt(chan_id_high-chan_id_low)
                    prob_spurious_line = None
                    line_set.append((num, sn_fin_line, sn_cor_line, chan_id_low+1, chan_id_high+1, freq[chan_id_low], freq[chan_id_high], vel[chan_id_low], vel[chan_id_high], line_width_for_others, flux_int_line, prob_spurious_line))


                    print('################### The snr of other line with consistent z! ####################')
                    print(str(line_info['line_name'][i])+' is in this freq range! It is at '+str(tar_freq)+'GHz, the S/N is '+str(sn_cor_line))
                    print('#################################################################################')


        ## plot the fig of spectrum
        figure, ax = plt.subplots(figsize=(20,4))
        ax.errorbar(freq-np.abs(freq[0]-freq[1])/2,flux,yerr=eflux,marker='+',color='k',markersize=0.1,lw=0.5,linestyle='none')
        ax.step(freq,flux,color='k',lw=0.5)
        ax.plot([freq_lo,freq_lo],[1.5,13.3],'-',color='tab:orange',lw=0.5)
        ax.plot([freq_hi,freq_hi],[1.5,13.3],'-',color='tab:orange',lw=0.5)
        ax.fill_between([freq_lo,freq_hi],[flux.min()-1,flux.min()-1],[flux.max()+3,flux.max()+3],alpha=0.15,color='tab:orange')
        ax.plot([freq[0],freq[-1]],[0,0],'--',color='k',lw=0.5)
        ax.set_ylim([flux.min()-0.5*flux.max(),2*flux.max()])
        ax.set_xlim([freq[0],freq[-1]])
        ax.set_xlabel(r'Observing Frequency (GHz)',fontsize=12)
        ax.set_ylabel(r'Flux density (mJy)',fontsize=12)
        ax.text(0.5,0.93,r'S/N=%.1f' % snr_cor, transform=ax.transAxes, fontsize=10)
        ax.text(0.5,0.86,r'f$_{\rm obs}$=%.3f GHz' % freq_middle, transform=ax.transAxes, fontsize=10)
        ax.text(0.75,0.93,r'line-width=%i km/s' % line_width, transform=ax.transAxes, fontsize=10)
        ax.text(0.75,0.86,r'prob$_{\rm spurious}$=%.2e' % prob_spurious, transform=ax.transAxes, fontsize=10)
        
        ## add the possible consistent line, colored by green
        if snr_other_lines:
            i=0
            for chan_low,chan_high,sn_line in other_lines_info:
                if freq[chan_low]<freq_middle and freq_middle<freq[chan_high]:
                    continue
                else:
                    i+=1
                    ax.fill_between([freq[chan_low],freq[chan_high]],[flux.min()-1,flux.min()-1],[flux.max()+3,flux.max()+3],alpha=0.15,color='tab:green')
                    ax.text(0.5,0.86-0.07*i,r'S/N of other line (same z) = %.1f' % sn_line, transform=ax.transAxes, fontsize=10)

        ax.tick_params(which='both',direction='in')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        figure.savefig(f'{out_base}_probable_line_in_'+str(round(freq[0]))+'_'+str(round(freq[-1]))+'.pdf')


    out_path = f'{out_base}_snmax_output.txt'

    with open(out_path, 'w') as output:
        for line in line_set:
            num, sn, snr_cor, vi1, vj1, freq_lo, freq_hi, vlo, vhi, line_width, flux_int, prob_spurious = line
            if prob_spurious is None:
                prob_str = 'None'
            else:
                prob_str = f'{prob_spurious:.3e}'
            output.write(f'{int(num)} {sn:.5f} {snr_cor:.5f} {int(vi1):5d} {int(vj1):5d} {freq_lo:11.6f} {freq_hi:11.6f} {vlo:12.4f} {vhi:12.4f} {line_width:8.1f} {flux_int:8.2f} {prob_str}\n')



def main():

    home_dir = os.path.expanduser("~")
    
    parser = argparse.ArgumentParser(description='Detect the emission line in the spectrum')
    parser.add_argument('--path', type=str, required=True, help='The parent path of the target')
    parser.add_argument('--target', type=str, required=True, help='The dir name of target')
    parser.add_argument('--input-file', type=str, 
                        default='*.result.txt',
                        help='The pattern of input spectrum files, default: *.result.txt')
    parser.add_argument('--vel-low', type=int, default=200, help='Lower limit of the vel_width of a line')
    parser.add_argument('--vel-high', type=int, default=500, help='Higher limit of the vel_width of a line')
    parser.add_argument('--snr-other-lines', choices=['true', 'false'], default='false', help='whether calc the S/N for other lines')
    parser.add_argument('--line-info', type=str, 
                        default=os.path.join(home_dir, 'Software/Gildas_my_tool', 'config', 'line_candidates'),
                        help='The file of frequency of line candidates, help to compute the S/N for consistent lines')
    parser.add_argument('--redshift', type=float, help='The redshift you want to test')
    
    args = parser.parse_args()
    snr_other_lines = args.snr_other_lines.lower() == 'true'

    target_dir = os.path.join(args.path, args.target, 'uv_spec')
    if not os.path.exists(target_dir):
        print(f"Error: the directory {target_dir} does not exist")
        sys.exit(1)
        
    
    if not os.path.exists(args.line_info):
        print(f"Error: the line info file {args.line_info} does not exist")
        sys.exit(1)
    
    os.chdir(target_dir)
    input_pattern = args.input_file
    files = glob.glob(input_pattern)
    files.sort()
    
    if not files:
        print(f"Error: do not find matched {input_pattern} in {target_dir}")
        sys.exit(1)
     
    
    print(f'############ Now process {args.target} ! ###############')
    for fpath in files:
        line_detect(
            path=args.path,
            src=args.target,
            file=fpath,
            vel_width_low=args.vel_low,
            vel_width_high=args.vel_high,
            snr_other_lines=snr_other_lines,
            line_info_file=args.line_info,
            guess_redshift=args.redshift
        )

if __name__ == "__main__":
    main()
        

