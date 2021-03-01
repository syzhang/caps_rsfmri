# submit jobs

# extract time series
for f in ../data/func/*.nii.gz;  
    do 
    echo "extracting time series from ${f}"; 
    fsl_sub -T 2 -R 32 python extract_ts.py ${f}
done;

# dcc
# for f in ../output/yeo/*.npy; 
# for f in ../output/msdl/*.npy; 
for f in ../output/fan/*.npy; 
    do 
    echo "calculating dcc from ${f}"; 
    fsl_sub -T 20 -R 32 python correlations.py ${f}
done;
