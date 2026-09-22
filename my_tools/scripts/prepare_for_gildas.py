import shutil
import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.coordinates import SkyCoord
import astropy.units as u
import pdb


line_info_dict = {'J0055+0146': {'line_cen': [271.291677], 'FWHM': [0.333233993]},
 'J0100+2802': {'line_cen': [259.380181],
  'FWHM': [0.402239],
  'FWHM_for_con': 0.46},
 'J0109-3047': {'line_cen': [243.963076, 243.944505, 243.959424],
  'FWHM': [0.263702, 0.319626, 0.240664]},
 'J0129-0035': {'line_cen': [280.362341, 280.363793],
  'FWHM': [0.185613, 0.191214]},
 'J0142-3327': {'line_cen': [259.017781, 259.012885],
  'FWHM': [0.322615, 0.288967]},
 'J0210-0456': {'line_cen': [255.700587], 'FWHM': [0.148511]},
 'J0305-3150': {'line_cen': [249.608655, 249.607603, 249.59226, 249.61324],
  'FWHM': [0.183052, 0.231071, 0.243279, 0.189605],
  'FWHM_for_con': 0.5},
 'J0454-4448': {'line_cen': [269.263113], 'FWHM': [0.33538]},
 'J0842+1218': {'line_cen': [268.633987, 268.599625],
  'FWHM': [0.279183, 0.348381],
  'FWHM_for_con': 0.7},
 'J0859+0022': {'line_cen': [257.16218], 'FWHM': [0.351407]},
 'J1030+0524': {'line_cen': [260.374387], 'FWHM': [0.11376]},
 'J1044-0125': {'line_cen': [280.093734, 280.119531],
  'FWHM': [0.400387, 0.423186]},
 'J1048-0109': {'line_cen': [247.603869, 247.595588],
  'FWHM': [0.253643, 0.279094]},
 'J1120+0641': {'line_cen': [235.063158], 'FWHM': [0.37615]},
 'J1152+0055': {'line_cen': [258.084456, 258.082808],
  'FWHM': [0.158121, 0.251504]},
 'J1202-0057': {'line_cen': [274.260791, 274.274732],
  'FWHM': [0.291022, 0.30182]},
 'J1207+0630': {'line_cen': [270.098766], 'FWHM': [0.415531]},
 'J1208-0200': {'line_cen': [267.057365], 'FWHM': [0.226001]},
 'J1306+0356': {'line_cen': [270.20157, 270.219777],
  'FWHM': [0.210898, 0.257055]},
 'J1319+0950': {'line_cen': [266.433336, 266.43762],
  'FWHM': [0.490207, 0.466822]},
 'J1342+0928': {'line_cen': [222.552847], 'FWHM': [0.267121]},
 'J1509-1749': {'line_cen': [266.848881], 'FWHM': [0.536119]},
 'J2054-0005': {'line_cen': [269.999154, 269.993912],
  'FWHM': [0.216292, 0.236883]},
 'J2100-1715': {'line_cen': [268.410892, 268.393775],
  'FWHM': [0.239404, 0.333795],
  'FWHM_for_con': 0.7},
 'J2211-3206': {'line_cen': [258.97825], 'FWHM': [0.23398]},
 'J2216-0016': {'line_cen': [267.813411], 'FWHM': [0.284188]},
 'J2228+0152': {'line_cen': [268.430383], 'FWHM': [0.196218]},
 'J2229+1457': {'line_cen': [265.772124], 'FWHM': [0.271287]},
 'J2239+0207': {'line_cen': [262.152284], 'FWHM': [0.502798]},
 'J2310+1855': {'line_cen': [271.385015], 'FWHM': [0.361393]},
 'J2318-3029': {'line_cen': [265.983775, 265.958369],
  'FWHM': [0.261078, 0.234782]},
 'J2318-3113': {'line_cen': [255.338601, 255.325013],
  'FWHM': [0.287559, 0.212151]},
 'J2329-0301': {'line_cen': [256.265508], 'FWHM': [0.411541]},
 'J2348-3054': {'line_cen': [240.54654, 240.576647, 240.542898],
  'FWHM': [0.402877, 0.319283, 0.390987]},
 'PJ004+17': {'line_cen': [278.81], 'FWHM': [0.642117]},
 'PJ007+04': {'line_cen': [271.478814, 271.475844],
  'FWHM': [0.370563, 0.375239]},
 'PJ009-10': {'line_cen': [271.3157764], 'FWHM': [0.319155421]},
 'PJ011+09': {'line_cen': [254.450791], 'FWHM': [0.318613]},
 'PJ036+03': {'line_cen': [252.043706], 'FWHM': [0.192364]},
 'PJ056-16': {'line_cen': [272.812244], 'FWHM': [0.297855]},
 'PJ065-19': {'line_cen': [266.729702], 'FWHM': [0.32637]},
 'PJ065-26': {'line_cen': [264.428822, 264.456672, 264.466914],
  'FWHM': [0.318933, 0.379429, 0.310267]},
 'PJ158-14': {'line_cen': [268.867641], 'FWHM': [0.628694]},
 'PJ159-02': {'line_cen': [257.475406], 'FWHM': [0.30237]},
 'PJ167-13': {'line_cen': [252.91705, 252.888111, 252.903596],
  'FWHM': [0.44595, 0.385457, 0.431199]},
 'PJ183+05': {'line_cen': [255.500056, 255.501442],
  'FWHM': [0.314749, 0.313318]},
 'PJ217-16': {'line_cen': [265.898054], 'FWHM': [0.501437]},
 'PJ231-20': {'line_cen': [250.510312, 250.491677],
  'FWHM': [0.27981, 0.360212],
  'FWHM_for_con': 0.6},
 'PJ239-07': {'line_cen': [267.29507], 'FWHM': [0.421967]},
 'PJ308-21': {'line_cen': [262.675009, 262.660244],
  'FWHM': [0.50357, 0.501661],
  'FWHM_for_con': 0.7},
 'PJ323+12': {'line_cen': [250.483171], 'FWHM': [0.246344]},
 'PJ359-06': {'line_cen': [264.994921, 264.995838, 265.003486],
  'FWHM': [0.293394, 0.280374, 0.285892]},
 'VIMOS2911': {'line_cen': [265.837848], 'FWHM': [0.240265]},
 'J0136+0226':{'line_cen':[263.483577], 'FWHM':[0.172863]},
 'J0227-0605':{'line_cen':[263.597769], 'FWHM':[0.167749]},
 'J0909+0440':{'line_cen':[266.583288], 'FWHM':[0.229007]},
 'J1137+0045':{'line_cen':[257.976558], 'FWHM':[0.279387]},
 'J1146+0124':{'line_cen':[262.14619 ], 'FWHM':[0.269032]},
 'J1205-0000':{'line_cen':[246.101463], 'FWHM':[0.31914]},
 'J1217+0131':{'line_cen':[263.620688], 'FWHM':[0.445501]},
 'J1243+0100':{'line_cen':[235.362904], 'FWHM':[0.207625]},
 'J1406-0116':{'line_cen':[260.424396], 'FWHM':[0.225448]},
 'J2304+0045':{'line_cen':[258.52], 'FWHM':[0.32]},
 'J0244-5008':{'line_cen':[245.835014],'FWHM':[0.290521]},
 'J0038-1527':{'line_cen':[236.562], 'FWHM':[0.267239233]},
 'J0213-0626':{'line_cen':[246.14 ], 'FWHM':[0.238687285]},
 'J0218+0007':{'line_cen':[244.599], 'FWHM':[0.446675247]},
 'J0224-4711':{'line_cen':[252.656], 'FWHM':[0.281212057]},
 'J0229-0808':{'line_cen':[246.028], 'FWHM':[0.220542229]},
 'J0246-5219':{'line_cen':[240.951], 'FWHM':[0.321179235]},
 'J0252-0503':{'line_cen':[237.549], 'FWHM':[0.311101667]},
 'J0313-1806':{'line_cen':[219.914], 'FWHM':[0.172216951]},
 'J0319-1008':{'line_cen':[242.801], 'FWHM':[0.588225274]},
 'J0411-0907':{'line_cen':[242.848], 'FWHM':[0.300238521]},
 'J0430-1445':{'line_cen':[246.368], 'FWHM':[0.301305817]},
 'J0439+1634':{'line_cen':[252.759], 'FWHM':[0.238367557]},
 'J0525-2406':{'line_cen':[252.071], 'FWHM':[0.217559496]},
 'J0706+2921':{'line_cen':[249.95 ], 'FWHM':[0.343999193]},
 'J0910+1656':{'line_cen':[245.9  ], 'FWHM':[0.310565971]},
 'J0910-0414':{'line_cen':[248.883], 'FWHM':[0.649398269]},
 'J0921+0007':{'line_cen':[251.242], 'FWHM':[0.187540209]},
 'J0923+0402':{'line_cen':[248.988], 'FWHM':[0.290405695]},
 'J0923+0753':{'line_cen':[247.412], 'FWHM':[0.23827191]},
 'J1007+2115':{'line_cen':[223.191], 'FWHM':[0.259571986]},
 'J1058+2930':{'line_cen':[250.579], 'FWHM':[0.28056852]},
 'J1104+2134':{'line_cen':[244.718], 'FWHM':[0.541491763]},
 'J2002-3013':{'line_cen':[247.223], 'FWHM':[0.253741957]},
 'J2102-1458':{'line_cen':[247.967], 'FWHM':[0.180964186]},
 'J2211-6320':{'line_cen':[242.265], 'FWHM':[0.259149256]},
 'J0252-0503':{'line_cen':[237.557], 'FWHM':[0.21770017]},
 'J0837+4929':{'line_cen':[248.591], 'FWHM':[0.186391461]},
 'J1135+5011':{'line_cen':[250.563], 'FWHM':[0.354862384]},
 'PJ083+11':{'line_cen':[258.931466], 'FWHM':[0.177677]}}

def get_freqs_width_and_cord(name,vis):
    os.chdir('/disk3/high_z_QSOs/'+name)
    ## get the spw id and chan width!
    listfile = vis+'.listobs'
    if os.path.exists('/disk3/high_z_QSOs/'+name+'/'+listfile)==False:
        listobs(vis+'.ms',listfile=listfile)
    with open(listfile,'r') as listobs_file:
        all_lines = listobs_file.readlines()
    for index,item in enumerate(all_lines):
        if item.startswith('Fields:'):
            field_index = index
        if item.startswith('Spectral Windows:'):
            spw_index = index    
        if item.startswith('Sources:'):
            src_index = index       
            break
    field_lines =  all_lines[field_index+2]
    field_info = field_lines.split()
    for i in range(len(field_info)-1):
        if ':' in field_info[i]:
            ra_str = field_info[i]
            dec_str = field_info[i+1].replace(".", ":",2)
            break
    # 创建SkyCoord对象，并指定输入的天文坐标系统
    coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))

    # 获取RA和Dec的角秒表示
    ra_arcsec = coord.ra.to(u.arcsec).value
    dec_arcsec = coord.dec.to(u.arcsec).value

    spw_lines = all_lines[spw_index+2:src_index]
    spwid = []
    spw_width = []
    freqs = []
    
    for line in spw_lines:
        parts = line.split()
        spwid.append(eval(parts[0]))
        chanwid = eval(parts[5])/1000
        if chanwid>30:
            print('#### This is spw '+parts[0]+', it has strange channel width!! Check the listobs, then enter: chanwid=xxx !####')
            pdb.set_trace()
            
        spw_width.append(chanwid)
        
    for i in spwid:
        ## get the channel and frequency info for each spw!
        ms.open(vis+".ms")
        freq = ms.cvelfreqs(spwids=i,mode='channel',width=0,outframe='LSRK')/1E6
        freqs.append(freq)
        ms.close()
  
    return freqs,spw_width,ra_arcsec,dec_arcsec


def get_line_and_cont_channel(line_cen,FWHM_for_cont,FWHM_for_line,name,vis):
    os.chdir('/disk3/high_z_QSOs/'+name)
    ## get the spw id and chan width!
    listfile = vis+'.listobs'
    if os.path.exists('/disk3/high_z_QSOs/'+name+'/'+listfile)==False:
        listobs(vis+'.ms',listfile=listfile)
    with open(listfile,'r') as listobs_file:
        all_lines = listobs_file.readlines()
    for index,item in enumerate(all_lines):
        if item.startswith('Spectral Windows:'):
            spw_index = index    
        if item.startswith('Sources:'):
            src_index = index       
            break
    spw_lines = all_lines[spw_index+2:src_index]
    spwid = []

    for line in spw_lines:
        parts = line.split()
        spwid.append(eval(parts[0]))
        
    line_freq_l = line_cen - 1.5*FWHM_for_cont
    line_freq_r = line_cen + 1.5*FWHM_for_cont

    line_spw = []
    cont_spw = []
    line_map_spw = []


    ## search for line in all spws
    for i in spwid:
        ## get the channel and frequency info for each spw!
        ms.open(vis+".ms")
        freq = ms.cvelfreqs(spwids=i,mode='channel',width=0,outframe='LSRK')/1E6
        ms.close()
        nchan = len(freq)
        ## search for line
        spw_line_l,spw_line_r = np.argmin(np.abs(freq-line_freq_l)),np.argmin(np.abs(freq-line_freq_r))
        if spw_line_l==spw_line_r:
            print('############ This is spw '+str(i)+', and it does not contain line! ############')
            cont_spw.append(str(i)+':2~'+str(nchan-3))
        else:
            print('############ This is spw '+str(i)+', and it contains line! ############')
            ## get the channels for line cube and cont map
            line_l_id = min(spw_line_l,spw_line_r)
            line_r_id = max(spw_line_l,spw_line_r)
            if line_l_id==0:
                line_l_id = 2
            if line_r_id==nchan-1:
                line_r_id = nchan-3
            line_spw.append(str(i)+':'+str(line_l_id)+'~'+str(line_r_id))
            line_free = get_line_free_chan(i,line_l_id,line_r_id,nchan)
            if line_free!=None:
                cont_spw.append(line_free)
            ## get the channels for line map
            line_freq_for_map_l = line_cen - 1.2*FWHM_for_line
            line_freq_for_map_r = line_cen + 1.2*FWHM_for_line
            line_map_l,line_map_r = np.argmin(np.abs(freq-line_freq_for_map_l)),np.argmin(np.abs(freq-line_freq_for_map_r))
            line_map_l_id = min(line_map_l,line_map_r)
            line_map_r_id = max(line_map_l,line_map_r)
#             if line_map_l_id==0:
#                 line_map_l_id = 2
#             if line_map_r_id==nchan-1:
#                 line_map_r_id = nchan-3
            line_map_spw.append(str(i)+':'+str(line_map_l_id)+'~'+str(line_map_r_id))
    return line_spw,cont_spw,line_map_spw

def get_line_free_chan(spw,line_l,line_r,nchan):
    print('######### spw'+str(spw)+' has '+str(nchan)+' channels! ###########')
    if line_l<=3 and line_r<nchan-4:
        part = str(spw)+':'+str(line_r+1)+'~'+str(nchan-3)
    elif line_r>=nchan-4 and line_l>3:
        part = str(spw)+':2~'+str(line_l-1)
    elif line_l>3 and line_r<nchan-4:
        part = str(spw)+':2~'+str(line_l-1)+';'+str(line_r+1)+'~'+str(nchan-3)
    else:
        part=None
    return part
        

for src in srcs:
    if src.endswith('_high'):
        name = src[:-5]
    else:
        name = src
    print('########### start to process '+src+'! ###########')
    os.chdir('/disk3/high_z_QSOs/'+src)

    files = os.listdir('/disk3/high_z_QSOs/'+src)
    multi_proj_flag = False
    if len(set([file[:15] for file in files if file.endswith('kms.ms') or file.endswith('km.ms')]))>1:
        multi_proj_flag = True

    freqs,widths,ra,dec = get_freqs_width_and_cord(src,vis=src)

    if multi_proj_flag:
        ## avg line center
        line_cen = sum(line_info_dict[name]['line_cen'])/len(line_info_dict[name]['line_cen'])*1000
        ## max FWHM
        FWHM_for_line = max(line_info_dict[name]['FWHM'])*1000
        if 'FWHM_for_con' in line_info_dict[name].keys():
            FWHM_for_line_free = line_info_dict[name]['FWHM_for_con']*1000  ## wider FWHM to avoid compaions' emission line
        else:
            FWHM_for_line_free = FWHM_for_line
    else:
        line_cen = line_info_dict[name]['line_cen'][0]*1000
        FWHM_for_line = line_info_dict[name]['FWHM'][0]*1000
        if 'FWHM_for_con' in line_info_dict[name].keys():
            FWHM_for_line_free = line_info_dict[name]['FWHM_for_con']*1000
        else:
            FWHM_for_line_free = FWHM_for_line
    line_spw,cont_spw,line_map_spw = get_line_and_cont_channel(line_cen,FWHM_for_line_free,FWHM_for_line,src,vis=src)
    ##### make line ms ######
    if os.path.exists('/disk3/high_z_QSOs/'+src+'/contsub/')==False:
        os.mkdir('/disk3/high_z_QSOs/'+src+'/contsub/')

    for item in line_map_spw:
        spwid = eval(item.split(':')[0])
        line_chan = item.split(':')[1]
        start_chan = eval(line_chan.split('~')[0])
        end_chan = eval(line_chan.split('~')[1])
        line_free_chan = cont_spw[spwid]
        output_ms = '/disk3/high_z_QSOs/'+src+'/contsub/'+src+'_spw'+str(spwid)+'_contsub'
        # output_ms = src+'.ms.contsub'
        print('##### doing uvcontsub #####')
        print(item)
        # uvcontsub(src+'.ms',spw=str(spwid),fitspw=line_free_chan,fitorder=0)
        uvcontsub(src+'.ms',outputvis=output_ms,spw=str(spwid),fitspec=line_free_chan,fitorder=0) # casa-6.5.4
        default(split)
        #split(output_ms,outputvis=output_ms+'_line',spw=item,datacolumn='data') 
        split(output_ms,outputvis=output_ms+'_avg',spw=item,width=end_chan-start_chan+1,datacolumn='data') ## average the line channels part
    
    
    if os.path.exists('/disk3/high_z_QSOs/size_gildas/')==False:
        os.mkdir('/disk3/high_z_QSOs/size_gildas/')
    if os.path.exists('/disk3/high_z_QSOs/size_gildas/'+src)==False:
        os.mkdir('/disk3/high_z_QSOs/size_gildas/'+src)
        
    if os.path.exists('/disk3/high_z_QSOs/'+src+'/contsub/'+src+'_CII_1chan.ms'):
        shutil.rmtree('/disk3/high_z_QSOs/'+src+'/contsub/'+src+'_CII_1chan.ms')
        shutil.rmtree('/disk3/high_z_QSOs/'+src+'/contsub/'+src+'_CII.ms')
        files_in_contsub = os.listdir('/disk3/high_z_QSOs/'+src+'/contsub/')
        del_file = [file for file in files_in_contsub if '_CII_spw' in file]
        for file in del_file:
            shutil.rmtree('/disk3/high_z_QSOs/'+src+'/contsub/'+file)

    files_in_contsub = os.listdir('/disk3/high_z_QSOs/'+src+'/contsub/')
    concat_CII = ['/disk3/high_z_QSOs/'+src+'/contsub/'+file for file in files_in_contsub if file.endswith('_contsub_avg')]
    print('##### making CII ms file #####')
    concat(vis=concat_CII,concatvis='/disk3/high_z_QSOs/'+src+'/contsub/'+src+'_CII.ms', freqtol='1GHz')
    
    tb.open('contsub/'+src+'_CII.ms/SPECTRAL_WINDOW', nomodify=False)
    chan_width = tb.getcol('CHAN_WIDTH')[0]
    if len(chan_width)>1:
        tar_chan_width_ind = np.argmin(abs(chan_width))
        tar_chan_width = chan_width[tar_chan_width_ind]
        for i in range(len(chan_width)):
            if i!=tar_chan_width_ind:
                tb.putcell('CHAN_WIDTH', i, [tar_chan_width])
                tb.putcell('EFFECTIVE_BW', i, [abs(tar_chan_width)])
                tb.putcell('RESOLUTION', i, [abs(tar_chan_width)])
                tb.putcell('TOTAL_BANDWIDTH', i, abs(tar_chan_width))
            split('contsub/'+src+'_CII.ms',outputvis='contsub/'+src+'_CII_spw'+str(i)+'.ms',spw=i,datacolumn='data')            
        tb.close()
        concat(vis=['contsub/'+src+'_CII_spw'+str(i)+'.ms' for i in range(len(chan_width))], concatvis='contsub/'+src+'_CII_1chan.ms', freqtol='1GHz')
        export_file = 'contsub/'+src+'_CII_1chan.ms'
    else:
        tb.close()
        export_file = '/disk3/high_z_QSOs/'+src+'/contsub/'+src+'_CII.ms'

    export_file_copy = export_file[:-3]+'_copy.ms'
    tb.open(export_file+'/FEED', nomodify=False)
    a = tb.getcol('RECEPTOR_ANGLE')
    tb.close()
    copy_flag = False
    for i in range(len(a)):
        if len(set(a[i][:]))!=1:
            a[i][:]=0.0
            print('################ We need to make a copy file, and change RECEPTOR_ANGLE! ##################')
            copy_flag = True
    if copy_flag:
        os.system('cp -r '+export_file+' '+export_file_copy)
        tb.open(export_file_copy+'/FEED', nomodify=False)
        tb.putcol('RECEPTOR_ANGLE', a)
        tb.close()
        export_file = export_file_copy

    exportuvfits(export_file,fitsfile='/disk3/high_z_QSOs/size_gildas/'+src+'/CII.uvfits',overwrite=True) 



