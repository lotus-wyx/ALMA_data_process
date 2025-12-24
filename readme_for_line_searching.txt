Line detection
 1. run uv_fit in mapping, setting the para in that window
 2. after getting the xx.uvfit file, run export_uvfit.map file in mapping by 
    "@~/Software/Gildas_my_tool/scripts/export_uvfit.map xx", where xx is the file name.
 3. line searching through terminal:
    enter the dir containing the target folder, 
    e.g., here the current dir is "/disk2/Noema_high_z_galaxies/w24ed", 
    run "line_search_step3 -t 16042 -f '*.result.txt' -l 200 -h 600 -s true -z 3.98" ; 
        "16042" is the target name (also the folder name), 
        "true" means that we have a guess of z and now want to calc the snr for other consistent lines.
        "3.98" is the guess of z.

Check the output files:
the last col: 
   if it is a number -> P_spurious; 
   if it is None --> this line is selected due to input redshift, and it's S/N is calc with certain width

add 
如果给定红移对应的某个发射线的freq与line search发现的S/N最高的那个一致，
将其频率固定在那个line search的结果，并调整z，搜寻剩下的线，并将速度宽度定为这个线的结果

如果不一致，则将速度宽度定为300-400km/s