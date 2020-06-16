import numpy as np
from scipy.ndimage import interpolation
import pandas as pd

size = 28;
PixValMax=255.0 # use float value instead of int

### --- The following two functions of "moments" and "deskew" have been used from ....
### --- https://fsix.github.io/mnist/Deskewing.html

def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    # modified the following two lines from the original source
    deskew_img = interpolation.affine_transform(image,affine,offset=offset)
    return deskew_img.flatten()

### ---
### ---

# to read the .csv file for train and test data 
data = pd.read_csv('.\\orig_data\\mnist_train.csv', delimiter=',');
print (data.head());

# deskewing the pixel data for all the index
for i in range (0, data.count(1).size ):
    # create array from dataframe , size 28x28, normalize the value 
    image = np.array(data.iloc[i,1:]).reshape(size,size)/PixValMax
    # perform deskew and scale back to 255
    image = deskew(image)*PixValMax
    # clip the values between 0 to 255 to avaoid negative values
    image = np.clip( image, 0, 255 )
    # create 2D array and convert data type uint8
    deskewdata = np.array( [ image.astype(np.uint8) ] )
    # concatenate for non-zero index
    if (i == 0): deskewarr = deskewdata
    else: deskewarr = np.concatenate( (deskewarr, deskewdata), axis=0 )

    if (i%5000 == 0): print( "Data preprocessed ... ", i, " out of ", data.count(1).size)
        
# create Dataframe    
deskewDF = pd.DataFrame(deskewarr)

#add label column
deskewDF = pd.concat([data['label'],deskewDF], axis=1)
deskewDF.columns = data.columns;

# filename1 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S");
deskewDF.to_csv('.\\mod_data\\deskewdata_train.csv', index=False);

    


