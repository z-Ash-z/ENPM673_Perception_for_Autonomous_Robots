import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv


#Function for getting the coordinates of the higest and the lowest points of the ball
# Input: video read using cv.VideoCapture
# Output: The X and Y coordinates of the ball
def ball_coordinate_finder (ball_video):
    y_coordinates = []
    x_coordinates = []
    
    #Extracting the frames from the videos and getting the coordinates of the ball
    while ball_video.isOpened():
        ret, frame = ball_video.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
        
        #Filtering the red channel
        _, filtered_img = cv.threshold(frame[:, :, 2], 240, 255, cv.THRESH_BINARY)
        
        cv.imshow('frame', frame)
        cv.imshow('filtered image', filtered_img)
        
        #Finding the ball's locations from the filtered image
        ball_location = np.where(filtered_img < 255)
        ball_location = np.array(ball_location)
        highest_point = ball_location.max(1)
        lowest_point = ball_location.min(1)
        y_coordinates.append(highest_point[0])
        y_coordinates.append(lowest_point[0])
        x_avg = (highest_point[1] + highest_point[1]) / 2
        x_coordinates.append(x_avg)
        x_coordinates.append(x_avg)
                
        if cv.waitKey(1) == ord('q'):
            break

    ball_video.release()
    cv.destroyAllWindows()
    return x_coordinates, y_coordinates


#Fuction to find the standard least squares
# Input: the X and Y coordinates of the ball found in ball_coordinate_finder
# Output: the Y coordinates found using the standard least squares
def standard_least_squares (x_coordinates,y_coordinates):
    x_coordinates = np.asarray(x_coordinates)
    y_coordinates = np.asarray(y_coordinates)
    
    X = np.reshape(x_coordinates,(x_coordinates.shape[0],1))
    Y = np.reshape(y_coordinates,(y_coordinates.shape[0],1))
    n = X.shape[0]
    
    #Finding X^2 
    X_2 = np.square(X)
    
    #X_matrix = [X^2, X and 1's] 
    X_matrix = np.append(np.append(X_2, X, axis = 1), np.ones((n,1)), axis = 1)
    
    #Finding inverse of (X^T*X)
    XT_X_inv = np.linalg.inv(X_matrix.transpose().dot(X_matrix))
    
    #Finding X^T*Y
    XT_Y = (X_matrix.transpose()).dot(Y)
    
    #Finding B
    B = XT_X_inv.dot(XT_Y)
    
    #Calculating the Y using Y = X.B
    Y_corrected = X_matrix.dot(B)
    
    return Y_corrected

if __name__ == '__main__':

    ball_vid = []
    
    #Reading both the videos
    ball_vid.append(cv.VideoCapture('ball_video1.mp4'))
    ball_vid.append(cv.VideoCapture('ball_video2.mp4'))
    
    #Loops through both the videos for finding coordinates and fiiting the curves
    for i in range(0, len(ball_vid)):
    
        #Finding the coordinates and fitting the curves for ball video 1
        x_coordinates, y_coordinates = ball_coordinate_finder(ball_vid[i])
        Y_corrected = standard_least_squares(x_coordinates, y_coordinates)
        
        #Plotting the ball's coordinates and the fitted curve
        plt.plot(x_coordinates, y_coordinates, 'ro', label = 'ball {} coordinates'.format(i + 1))
        plt.plot(x_coordinates, Y_corrected, 'k-', label = 'ball {} fitted curve'.format(i + 1))
        plt.legend()
        plt.title('Ball {} coordinates and best fit curve'.format(i + 1))
        ax = plt.gca()
        ax.invert_yaxis()
        plt.show()