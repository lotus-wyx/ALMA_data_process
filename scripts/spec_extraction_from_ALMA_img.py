#!/usr/bin/env python3

import numpy as np
import copy
import astropy.units as u
import glob
import matplotlib.pyplot as plt
from   astropy.io          import fits
from   astropy             import wcs
# from   photutils           import aperture_photometry, CircularAperture
from   matplotlib.patches  import Ellipse
import random
from astropy.stats import sigma_clipped_stats

def extract_circle_aperture(cent_x,cent_y,freq_len,x_len,y_len,aperture_pix):
    ra_grid, dec_grid = np.meshgrid(np.arange(x_len), np.arange(y_len))
    distance_grid = np.sqrt((ra_grid - cent_x)**2 + (dec_grid - cent_y)**2)

    circle_mask = distance_grid <= aperture_pix
    mask_expanded = np.expand_dims(circle_mask, axis=(0, 1))
    spec_mask = np.repeat(mask_expanded, freq_len, axis=1)
    return distance_grid,spec_mask

def extract_spec(fits_f,n,target_ra,target_dec,result_file_path,aper_correct_flag=False): # n是aperture相对beam_maj/2的倍数
    with fits.open(fits_f) as hdul:
        print('### procrssing '+fits_f+' ########')
        # get ref ra and dec info
        data = hdul[0].data
        header = hdul[0].header

        ref_ra = header['CRVAL1']
        ref_dec = header['CRVAL2']
        delta_ra = header['CDELT1']
        delta_dec = header['CDELT2']
        cell = np.abs((header['cdelt1']* u.deg).to(u.arcsec).value)

        # get beam info
        beam_info = hdul[1].data
        beam_maj = np.array([beam_[0] for beam_ in beam_info])
        beam_min = np.array([beam_[1] for beam_ in beam_info])
        
        aperture_pix = np.max(beam_maj)/cell/2*n  ## aperture radius
        beam_area_pixel = np.pi*beam_maj*beam_min/4/np.log(2)/cell**2
        for i in range(data.shape[1]): # Jy/beam --> Jy
            data[0,i,:,:] = data[0,i,:,:]/beam_area_pixel[i]


        ra_values = ref_ra + (np.arange(data.shape[2]) - header['CRPIX1']) * delta_ra    # in deg
        dec_values = ref_dec + (np.arange(data.shape[3]) - header['CRPIX2']) * delta_dec # in deg

        ra_index = np.abs(ra_values - target_ra).argmin()
        dec_index = np.abs(dec_values - target_dec).argmin()
        
        print('######### START TO EXTEACT SPECTRUM! ##########')

        distance_grid,spec_mask = extract_circle_aperture(ra_index,dec_index,data.shape[1],data.shape[2],data.shape[3],aperture_pix)
        circle_spectrum = np.nansum(data*spec_mask, axis=(2, 3))
        
        ## sample the bkg noise (100 apertures)
        print('######### START TO EXTEACT SPECTRUM FROM BKG! ##########')
        bkg_ = distance_grid > aperture_pix*2
        bkg_ra,bkg_dec = np.where(bkg_==True)
        random_loc_ra = random.sample(list(bkg_ra),100)
        random_loc_dec = random.sample(list(bkg_dec),100)
        bkg_spectrum = []
        for i in range(len(random_loc_ra)):
            bkg_ra = random_loc_ra[i]
            bkg_dec = random_loc_dec[i]
            distance_grid_,bkg_mask = extract_circle_aperture(bkg_ra,bkg_dec,data.shape[1],data.shape[2],data.shape[3],aperture_pix)
            bkg_spectrum.append(np.nansum(data*bkg_mask, axis=(2, 3))[0])

        bkg_spectrum = np.array(bkg_spectrum)
        mean, bkg_flux, bkg_noise = sigma_clipped_stats(bkg_spectrum,axis=0)

        nchan = header['NAXIS3']
        ref_freq = header['CRVAL3']
        freq_step = header['CDELT3']
        ref_freq_chanid = header['CRPIX3']
        freq_list = np.array([ref_freq + freq_step * (i+1 - ref_freq_chanid) for i in range(nchan)])/10**9

        print('######## writing no_Apcor result for '+fits_f+' ###########')
        with open(result_file_path+'_no_Apcor.result.txt', "w") as r_file:
            r_file.write("#\tFrequency\tFlux\te_Flux\tbkg_flux\n")
            r_file.write("#\tGHz\tJy/beam\tJy/beam\tJy/beam\n")
            for freq, flux, e_flux, bkg_f in zip(freq_list, circle_spectrum[0], bkg_noise,bkg_flux):
                if freq<323 or freq>328:
                    r_file.write(f"\t{freq}\t{flux}\t{e_flux}\t{bkg_f}\n")




        if aper_correct_flag:
            psffile  = fits_f[:-5]+'.psf'
            #beamfile = fits_f[:-5]+'.fit_beam'
            beamfile = fits_f[:-5]+'_'+str(n)+'.fit_beam'
            region_beam_fit   =  'circle[['+str(data.shape[2]/2.)+'pix,' +str(data.shape[3]/2.)+'pix],' +str(np.max(beam_maj)*n)+'arcsec]'
            imfit(imagename=psffile, region=region_beam_fit,model=beamfile)
            exportfits(imagename=beamfile,fitsimage=beamfile+'.fits',overwrite=True)

            BeamHeader     =  imstat(beamfile) 
            beamimage      =  fits.open(beamfile+'.fits')[0]
            header_        =  beamimage.header
            image_         =  beamimage.data

            target_location_pix  = (BeamHeader['maxpos'][0], BeamHeader['maxpos'][1])

            Aper_Corr = image_[0,:,0,0]
            #f, ax1  = plt.subplots(1)

            for channel in range(0, BeamHeader['trc'][3]+1):
                beam       = image_[0,channel,:]
                radii      = np.arange(1,5,0.5)
                growth     = []
                fluxes     = []

                for radius in radii:
                    radius_pix    = radius / abs(header_['CDELT1'] * 3600)               ## in pixel
                    apertures     = CircularAperture(target_location_pix, r=radius_pix)
                    phot_table    = aperture_photometry(beam, apertures)
                    flux_aper     = phot_table['aperture_sum'].data
                    fluxes.append(flux_aper)

                mean   = np.mean(fluxes[-5:-1]) ## mean value of the last few fluxes (reach flat, could stand for the total flux)
                growth = fluxes / mean
                bad    = np.isnan(growth)
                growth[bad] = 1
                Aper_Corr[channel] = np.interp(np.max(beam_maj)*n/2., radii, growth[:,0]) ## 进行插值，计算出抽谱孔径实际应对应的孔径改正系数
                #ax1.plot(radii,growth)

            Flux_corrected = (circle_spectrum/Aper_Corr)[0]
            Flux_final = Flux_corrected
#             ax1.set_xlabel('Aperture radii [\"]')
#             ax1.set_ylabel('Power fraction')
#             ax1.set_title(r'Curve of growth for the clean beams (2-D Gaussian fitted)')
#             plt.savefig(fits_f[:-5]+'_aper_correction.pdf')
            print('######## writing Apcor result for '+fits_f+' ###########')
            with open(result_file_path+'_Apcor.result.txt', "w") as r_file:
                r_file.write("#\tFrequency\tFlux\te_Flux\tbkg_flux\n")
                r_file.write("#\tGHz\tJy/beam\tJy/beam\tJy/beam\n")
                for freq, flux, e_flux, bkg_f in zip(freq_list, Flux_final, bkg_noise/Aper_Corr,bkg_flux/Aper_Corr):
                    if freq<323 or freq>328:
                        r_file.write(f"\t{freq}\t{flux}\t{e_flux}\t{bkg_f}\n")



image_file = '/disk2/H-dropout/2021.1.01650.S/2nd_try_with_dliu_pipe/Each_target_img/COS-25363/datacubes/COS-25363'
exportfits(image_file+'.image', fitsimage=image_file+'.fits',overwrite=True)

fits_f = image_file+'.fits'
target_ra,target_dec = 150.1111333,2.5239389
for n_try in [0.5,1,1.5,2]: #0.5,1,
    print("############### Having test for "+str(n_try)+" times beam ##########")
    result_file_path = image_file+'_'+str(n_try)
    extract_spec(fits_f,n_try,target_ra,target_dec,result_file_path,aper_correct_flag=False)


