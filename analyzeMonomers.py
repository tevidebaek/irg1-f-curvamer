import numpy as np
import scipy.ndimage as ndim
import matplotlib.pyplot as plt
import os
import pims
from PIL import Image
import skimage.morphology

def fit_circle_least_squares(points):
    # Building the design matrix
    A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
    B = np.sum(points**2, axis=1)
    C = np.linalg.lstsq(A, B, rcond=None)[0]
    
    # Circle parameters
    xc = C[0] / 2
    yc = C[1] / 2
    radius = np.sqrt(C[2] + xc**2 + yc**2)
    
    return xc, yc, radius

def calculate_arc_length(points, xc, yc, radius):
    # Calculate the angles subtended by the points at the center
    angles = np.arctan2(points[:,1] - yc, points[:,0] - xc)
    
    # Ensure the angles are in the range [0, 2*pi]
    angles = np.mod(angles, 2*np.pi)
    
    # Sort the angles
    angles = np.sort(angles)
    
    # Calculate the differences between consecutive angles
    angle_diffs = np.diff(angles)
    
    # Include the wrap-around difference
    angle_diffs = np.append(angle_diffs, 2*np.pi - angles[-1] + angles[0])
    
    # Find the largest angle difference (indicating the smallest arc)
    max_angle_diff = np.max(angle_diffs)
    
    # Calculate the arc length of the complementary angle
    arc_length = radius * (2*np.pi - max_angle_diff)
    
    return arc_length

#load images into memory


image_src = './ExtractedImages/Monomer/'
image_dir = './'+image_src+'/'

#before looping through the images open a file

f = open('monomer_properites.txt', "w")
f.write("Img_name, Radius, Arc_length \n")

for dirpath, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.endswith('.tif'):
                image_file = pims.ImageSequence(image_dir + filename)[0]
     
                #plt.figure()
                #plt.imshow(image.T, cmap='gray')
                #plt.axis('off')
                
                temp_image = np.copy(image_file).astype(float)

                if True:
                    #apply edge detection to the image
                    edge_filtered = np.sqrt(ndim.sobel(temp_image, 0)**2 + ndim.sobel(temp_image, 1)**2)
                    temp_image = edge_filtered

                    #plt.figure()
                    #plt.imshow(edge_filtered)

                if True:
                    #first apply a guassian blur to the image
                    gauss_image = ndim.gaussian_filter(temp_image, sigma=4)
                    temp_image = gauss_image
                    #plt.figure()
                    #plt.imshow(temp_image)

                if True:
                    #first adjust the dynamic range of the iamge
                    temp_image = (temp_image-np.min(temp_image))/(np.max(temp_image) - np.min(temp_image))*255.
                    #now apply a threshold to the image

                    threshold = 120
                    thres_image = np.copy(temp_image)
                    thres_image[thres_image>threshold]=255
                    thres_image[thres_image<=threshold]=0
                    temp_image = thres_image
                    #plt.figure()
                    #plt.imshow(temp_image)


                if True:
                    #now look at the largest region of the thresholded image
                    label_im, nb_labels = ndim.label(temp_image)
                    sizes = ndim.sum(temp_image, label_im, range(nb_labels +1))
                    mask = sizes==np.max(sizes)

                    largest_region = mask[label_im]

                    #plt.figure()
                    #plt.imshow(largest_region)

                    #get the skeleton of the image
                    skeleton = skimage.morphology.skeletonize(largest_region)

                    #plt.figure()
                    #plt.imshow(skeleton)

                    curvamer_points = np.array(np.where(skeleton)).T

                    #print(curvamer_points)

                    xc, yc, radius = fit_circle_least_squares(curvamer_points)

                    arc_length = calculate_arc_length(curvamer_points, xc, yc, radius)

                    theta = np.linspace(0, np.pi*2, 100)

                    circle_x = xc + radius * np.cos(theta)
                    circle_y = yc + radius * np.sin(theta)

                    #plt.figure()
                    #plt.imshow(image.T, cmap='gray')
                    #plt.plot(circle_x, circle_y, 'b--', alpha = 0.5)
                    #plt.scatter(curvamer_points[:,0], curvamer_points[:,1], c='r', alpha=0.04)
                    #plt.axis('off')

                    print(radius, arc_length)
                    f.write(filename+', '+str(radius)+', '+str(arc_length)+' \n')

                #plt.show()

f.close()