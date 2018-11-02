# make sure u are in the root directory when executing this script 
cd distances/dtw
rm __init__.py
rm dtw.c 
rm *.so 
rm -r build
rm -r __pycache__
python3 setup.py build_ext --inplace
touch __init__.py 
chmod 777 __init__.py 
# for shapeDTW
cd ../shapeDTWefficient
rm __init__.py
rm shapeDTWefficient.c
rm *.so
rm -r build
rm -r __pycache__
python3 setup.py build_ext --inplace
touch __init__.py
chmod 777 __init__.py
