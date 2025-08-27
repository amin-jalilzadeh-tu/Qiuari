import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Now run the test
exec(open('simple_uncertainty_test.py').read())