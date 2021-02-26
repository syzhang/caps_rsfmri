# submit jobs

# extract time series
for f in ../data/func/*.nii.gz;  
    do 
    echo "extracting time series from ${f}"; 
    fsl_sub -T 2 -R 32 python extract_ts.py ${f}
done;