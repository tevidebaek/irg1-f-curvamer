import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pims
from PIL import Image

'''
Extracting the positions of partilces from the Curvamer image folders
'''

position_files = 'CoordinateFiles_monomers.csv'
output_folder = "ExtractedImages/"

def getImageSections(position, outfn_src):
     
    cnt=1

    part_positions = pd.read_csv(position, delimiter=',')

    print(part_positions.head())

    #now that we have a list of the images, we can loop through the CSV file to get the particle positons

    subimage_size = 450 #this is the image size at x11000 mag
    mag_ref = 11000.
    x_bound, y_bound = 2560, 2048

    #break the parsing up into sections that have the same folder for images
    unique_folders = np.array(pd.unique(part_positions['Folder']))

    for folder in unique_folders:
        df_folder = part_positions[part_positions['Folder']==folder]

        #load all the images for this sub folder
        image_src = folder
        image_dir = './'+image_src+'/'

        images = []

        for dirpath, dirnames, filenames in os.walk(image_dir):
                for filename in filenames:
                    if filename.endswith('.tif'):
                        image_file = pims.ImageSequence(image_dir + filename)[0]
                        images.append(image_file)

        #now go through the elements of this sub dataframe to extract all the images

        for index, row in df_folder.iterrows():
        
            stack_pos = row['Stack']

            img_slice = int(stack_pos-1)

            x, y = row['X'], row['Y']

            magnification = float(row['Mag'])

            temp_subimage_size = subimage_size*magnification/mag_ref

            print(img_slice)

            x_left = int(np.max([0, x-temp_subimage_size/2]))
            x_right= int(np.min([x_bound, x+temp_subimage_size/2]))
            
            y_left = int(np.max([0, y-temp_subimage_size/2]))
            y_right= int(np.min([y_bound, y+temp_subimage_size/2]))

            print('x', x_left, x_right)
            print('y', y_left, y_right)

            cropped_image = images[img_slice][y_left:y_right, x_left:x_right]

            #plt.figure()
            #plt.imshow(cropped_image)
            #plt.show()

            im = Image.fromarray(cropped_image)

            #for the filename we will call it by the cnt as well as the temp and MgCl2 column value
            filename = 'T'+str(row['Temp'])+'_M'+str(row['MgCl2'])+'_'+str(cnt).zfill(4)+'.tif'

            im.save(outfn_src+'/'+filename)

            cnt+=1

getImageSections(position_files, output_folder)
