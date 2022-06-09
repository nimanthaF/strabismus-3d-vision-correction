#      ▄▀▄     ▄▀▄
#     ▄█░░▀▀▀▀▀░░█▄
# ▄▄  █░░░░░░░░░░░█  ▄▄
#█▄▄█ █░░▀░░┬░░▀░░█ █▄▄█

#######################################
##### Authors:                    #####
##### Stephane Vujasinovic        #####
##### Frederic Uhrweiller         ##### 
#####                             #####
##### Creation: 2017              #####
##### Optimization: David Castillo#####
##### Rv: FEB:2018                #####
#######################################


#***********************
#**** Main Programm ****
#***********************

# Package importation
from itertools import filterfalse
import time
import numpy as np
import cv2
import os
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from multiprocessing import Pool

# =========================sub Process===========================


# Filtering
kernel= np.ones((3,3),np.uint8)

def doWork(st): #j=1 es izquierdo , j=2 es derecho
    grayL = st[1] 
    grayR = st[2]
    j = st[2]

    # Create StereoSGBM and prepare all parameters
    window_size = 5
    min_disp = 2
    num_disp = 130-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 5,
        preFilterCap = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2)

    # Used for the filtered image
    if j == 1 :
        disp= stereo.compute(grayL,grayR)
    
    if j == 2 :
        stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time
        disp= stereoR.compute(grayR,grayL)

    return disp

#====================================================

cv2.useOptimized()
wb=Workbook()
ws=wb.active  

# write into the excell worksheet

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        """
				p p p
				p p p
				p p p
        """
        average=0
        for u in range (-1,2):     # (-1 0 1)
            for v in range (-1,2): # (-1 0 1)
                average += disp[y+u,x+v]
        average=(average/9)/5
        # Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        # Distance= np.around(Distance*0.01,decimals=2)
        # print('Distance: '+ str(Distance)+' m')
        print('distance: '+ str(average) + ' cm')
        # counterdist = int(input("ingresa distancia (cm): "))
        # ws.append([counterdist, average])

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

##===========================================================
filenameL = os.path.join("{}.npy".format("imgpointsL"))
filenameR = os.path.join("{}.npy".format("imgpointsR"))
filename_op = os.path.join("{}.npy".format("objpoints"))
filename_mtR = os.path.join("{}.npy".format("mtxR"))
filename_dR = os.path.join("{}.npy".format("distR"))
filename_mtL = os.path.join("{}.npy".format("mtxL"))
filename_dL = os.path.join("{}.npy".format("distL"))
filename_chR = os.path.join("{}.npy".format("ChessImaR"))

# Read
imgpointsR = np.load(filenameR)
imgpointsL = np.load(filenameL)
objpoints = np.load(filename_op)
mtxR = np.load(filename_mtR)
distR = np.load(filename_dR)
mtxL = np.load(filename_mtL)
distL = np.load(filename_dL)
ChessImaR = np.load(filename_chR)

print('Cameras Ready to use')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC


retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL,  RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped

# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)


#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 5
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    preFilterCap = 5,
    P1 =8*3*window_size**2,        
    P2 = 32*3*window_size**2)

# stereo=cv2.StereoBM_create(numDisparities = num_disp,blockSize = window_size)




# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000#80000
sigma = 1.8 #1.8
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#*************************************
#***** Starting the StereoVision *****
#*************************************
# Call the two cameras
CamR= cv2.VideoCapture(0)   # Wenn 0 then Right Cam and wenn 2 Left Cam
CamL= cv2.VideoCapture(1)

# def write_ply(vertices, colors, filename):
# 	colors = colors.reshape(-1,3)
# 	vertices = np.hstack([vertices.reshape(-1,3),colors])

# 	ply_header = '''ply
# 		format ascii 1.0
# 		element vertex %(vert_num)d
# 		property float x
# 		property float y
# 		property float z
# 		property uchar red
# 		property uchar green
# 		property uchar blue
# 		end_header
# 		'''
# 	with open(filename, 'w') as f:
# 		f.write(ply_header %dict(vert_num=len(vertices)))
# 		np.savetxt(f,vertices,'%f %f %f %d %d %d')

def write_ply(fn, verts, colors):
    ply = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def pointcloud(depth, fov):
        fy = fx = 0.5 / np.tan(fov * 0.5) # assume aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)
        
        x = mask[1]
        y = mask[0]
        
        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height
        
        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T

while True:
   
        #mark the start time
        startTime = time.time()
        # Start Reading Camera images
        retR, frameR= CamR.read()
        retL, frameL= CamL.read()

        # cv2.imshow("frameR", frameR)

        # cv2.imshow("frameL",frameL)    
        # cv2.imshow("frameR",frameR)     
        alpha = 0.6 # Simple contrast control
        beta = 0    # Simple brightness control

        
        # for y in range(frameR.shape[0]):
        #     for x in range(frameR.shape[1]):
        #         for c in range(frameR.shape[2]):
        #             frameR[y,x,c] = np.clip(alpha*frameR[y,x,c] + beta, 0, 255)

        # for y in range(frameL.shape[0]):
        #     for x in range(frameL.shape[1]):
        #         for c in range(frameL.shape[2]):
        #             frameL[y,x,c] = np.clip(alpha*frameL[y,x,c] + beta, 0, 255)

        
        # Rectify the images on rotation and alignement
        # Rectify the image using the calibration parameters founds during the initialisation
        Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  
        Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # Convert from color(BGR) to gray
        #grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        #grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

        grayR= Right_nice
        grayL= Left_nice

        # grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        # grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
        #=======================================================================================
        
        # Compute the 2 images for the Depth_image
        # Run the pool in multiprocessing
        ##st2 = (grayL,grayR,2 )

        


        # Computo para el stereo
        disp= stereo.compute(grayL,grayR) #.astype(np.float32)/ 16
        dispL= disp
        dispR= stereoR.compute(grayR,grayL)        


        #=======================================================================================
    
        dispL= np.int16(dispL)
        dispR= np.int16(dispR)
        
        # Using the WLS filter
        filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)

        # Filtering the Results with a closing filter
        closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)


        # Colors map
        dispc= (closing-closing.min())*255
        dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
        disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)  
        # Change the Color of the Picture into an Ocean Color_Map
        filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 

        h, w = frameL.shape[:2]
        f = 0.8*w                          # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5*w],
                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                [0, 0, 0,     -f], # so that y-axis looks up
                [0, 0, 1,      0]])

        mask=disp >disp.min()
        points=cv2.reprojectImageTo3D(disp,Q)
        colors = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)
        out_points = points[mask]
        out_colors = colors[mask]

        ######################################
        #filter by dimension
        idx = np.fabs(out_points[:,0]) < 4.5
        out_points = out_points[idx]
        out_colors = out_colors.reshape(-1, 3)
        out_colors = out_colors[idx]
        ######################################

        out_fn = 'out_4.ply'
        # cloud_plot=PyntCloud.from_file("out_1.ply")
        # cloud_plot.plot()

    
        if cv2.waitKey(1) & 0xFF == ord('s'):
            write_ply(out_fn,out_points, out_colors)
            print("point cloud saved!")
        

        # filt_Color_rotated= cv2.rotate(filt_Color, cv2.ROTATE_180)
        cv2.imshow('Filtered Color Depth',filt_Color)

        pcl=pointcloud(filteredImg,60)
        np.savetxt('recon2.ply',pcl)

        # # reflect on x axis
        # reflect_matrix = np.identity(3)
        # reflect_matrix[0] *= -1

        # reflected_pts = np.matmul(out_points, reflect_matrix)
        # projected_img,_ = cv2.projectPoints(reflected_pts, np.identity(3), np.array([0., 0., 0.]), mtxR, np.array([0., 0., 0., 0.]))
        # projected_img = projected_img.reshape(-1, 2)

        # def showImg(img):
        #     # plt.figure(figsize=(30, 30))
        #     plt=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     cv2.imshow('plot',plt)

        # blank_img = np.zeros(frameL.shape, 'uint8')
        # img_colors = frameL[mask][idx].reshape(-1,3)

        # for i, pt in enumerate(projected_img):
        #     pt_x = int(pt[0])
        #     pt_y = int(pt[1])
        #     if pt_x > 0 and pt_y > 0:
        #         # use the BGR format to match the original image type
        #         col = (int(img_colors[i, 2]), int(img_colors[i, 1]), int(img_colors[i, 0]))
        #         cv2.circle(blank_img, (pt_x, pt_y), 1, col)

        # showImg(blank_img)

        # Draw Red lines
        for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
            Left_nice[line*20,:]= (0,0,255)
            Right_nice[line*20,:]= (0,0,255)
      
        # Left_nice_rotated= cv2.rotate(Left_nice, cv2.ROTATE_180)
        # Right_nice_rotated= cv2.rotate(Right_nice, cv2.ROTATE_180)
        cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
        
        # Mouse click
        cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)
        
        #mark the end time
        endTime = time.time()    
        
        
        #calculate the total time it took to complete the work
        workTime =  endTime - startTime
        
        #print results
    #print ("The job took " + str(workTime) + " sconds to complete")

        # End the Programme
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    
# Save excel
wb.save("readvsdist.xlsx")

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
