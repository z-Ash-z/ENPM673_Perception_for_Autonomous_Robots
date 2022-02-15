import matplotlib.pyplot as plt
import numpy as np
import random

#Function to nomalize the given array
# Input: array X
# Output: nomalized array X
def normalization(x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)))
    
    

#Function to calculate the covariance between two sets of data
# Inputs: two data sets
# Output: covariance between the two data sets
def covriance(x, y):
    x_mean, y_mean = x.mean(), y.mean()
    return ((np.sum((x - x_mean) * (y - y_mean))) / (len(x) - 1))


#Function to find the covariance matrix
# Input: matrix x
# Output: Covariance matrix of matrix x
def find_covar_mat(x):
    return np.array([[covriance(x[0], x[0]), covriance(x[0], x[1])],
                     [covriance(x[1], x[0]), covriance(x[1], x[1])]])
   
   
#Function for plotting the projections of the directions of the greatest variance in data
# Input: Mean of x and y axises and covariance matrix
# Output: Plot of the the directions of the greatest variance in data
def plot_eigen_vals_vect(x_mean, y_mean, cov_mat):
    e_values, e_vectors = np.linalg.eig(cov_mat)
    e_vector_1 = e_vectors[:, 0]
    e_vector_2 = e_vectors[:, 1]
    print("\nEigen Values:\n", e_values)
    print("\nEigen Vectors:\n",e_vector_1,e_vector_2)
    plt.quiver(x_mean, y_mean, e_vector_1[0], e_vector_1[1],
               color = ['g'], scale = 1 / np.sqrt(e_values[0]))
    plt.quiver(x_mean, y_mean, e_vector_2[0], e_vector_2[1],
               color = ['b'], scale = 1 / np.sqrt(e_values[1]))
   
   
#Fuction to find the standard least squares
# Input: the X and Y coordinates
# Output: the Y coordinates found using the standard least squares
def standard_least_squares (x_coordinates, y_coordinates):
    x_coordinates = np.asarray(x_coordinates)
    y_coordinates = np.asarray(y_coordinates)
    
    X = np.reshape(x_coordinates,(x_coordinates.shape[0],1))
    Y = np.reshape(y_coordinates,(y_coordinates.shape[0],1))
    n = X.shape[0]
    
    #X_matrix = [X, 1's] 
    X_matrix = np.append(X, np.ones((n,1)), axis = 1)
    
    #Finding inverse of (X^T*X)
    XT_X_inv = np.linalg.inv(X_matrix.transpose().dot(X_matrix))
    
    #Finding X^T*Y
    XT_Y = (X_matrix.transpose()).dot(Y)
    
    #Finding B
    B = XT_X_inv.dot(XT_Y)
    
    #Calculating the Y using Y = X.B
    Y_corrected = X_matrix.dot(B)
    
    return Y_corrected
     
     
#Function to find the total least squares
# Input: the X and Y coordinates
# Output: the X and Y coordinates found using the standard least squares
def total_least_squares (x_coordinates, y_coordinates):
    x_mean, y_mean = np.mean(x_coordinates), np.mean(y_coordinates)
    
    U = np.sum((y_coordinates - y_mean)**2) - np.sum((x_coordinates - x_mean)**2)
    N = 2 * (np.sum((x_coordinates - x_mean).dot(y_coordinates - y_mean)))
    
    b = (U + (np.sqrt((U**2) + (N**2)))) / N
    a = y_mean - b * x_mean
    
    return x_coordinates, x_coordinates * b + a
    

#Function to find the distance between a point and a line
# Input: point, the points on the line
# Output: distance between point and the line
def distance(point, line_point_1, line_point_2):
    return abs(((line_point_2[1] - line_point_1[1]) * point[0]) - ((line_point_2[0] - line_point_1[0]) * point[1]) + (line_point_2[0] * line_point_1[1])
               - ([line_point_2[1] * line_point_1[0]])) / (np.sqrt((line_point_2[1] - line_point_1[1]) ** 2 + (line_point_2[0] - line_point_1[0]) ** 2))


#Function for fitting a line using RANSAC
# Input: x and y coordinates of the data and the distance threshold
# Output: X and Y coordinates of points on the best line and  X and Y coordinates of all inliers
def ransac(x_coordinates, y_coordinates, t):
    n = 1000
    sample = 0
    line_coordinates = []
    points_on_line = []

    while (sample < n):
        rand = random.sample(range(len(x_coordinates)), 2)
        p1 = (x_coordinates[rand[0]], y_coordinates[rand[0]])
        p2 = (x_coordinates[rand[1]], y_coordinates[rand[1]])
        inliers = []

        for i in range(len(x_coordinates)):
            if distance([x_coordinates[i], y_coordinates[i]], p1, p2) <= t:
                inliers.append([x_coordinates[i], y_coordinates[i]])

        if len(points_on_line) < len(inliers):
            points_on_line = inliers
            line_coordinates = [(p1[0], p2[0]), (p1[1], p2[1])]

        e = 1-(len(inliers)/len(x_coordinates))
        n = (np.log(e))/(np.log(1-(1-e)**2))
        sample += 1

    x_in = []
    y_in = []
    for i in points_on_line:
        x_in.append(i[0])
        y_in.append(i[1])
    return (line_coordinates[0], line_coordinates[1], x_in, y_in)
    

if __name__ == '__main__':
    #Reading the csv and getting the data
    with open('hw1_linear_regression_dataset.csv') as csv_file:
        rows = csv_file.readlines()
        
        age = []
        cost = []
        
        for row in rows:
            #print(row)
            temp = row.replace("\n", "").split(',')
            age.append(float(temp[0]))
            cost.append(float(temp[-1]))
    #print(age, cost)
    
    #Normalizing the X and Y axis for finding the Covariance matrix
    X_norm = np.array(normalization(age))
    Y_norm = np.array(normalization(cost))
    
    covar_mat = find_covar_mat(np.vstack((X_norm, Y_norm)))
    print('\nThe Covariance Matrix:\n', covar_mat)
    
    #Plotting the Principle Component Analysis
    plt.figure()
    plt.title('Principle Component Analysis')
    plt.plot(X_norm, Y_norm, 'ro')
    plt.xlabel('Ages')
    plt.ylabel('Insurance Cost')
    plot_eigen_vals_vect(np.mean(X_norm), np.mean(Y_norm), covar_mat)
    plt.show()
    
    #Plotting Linear least square method
    Y_corrected = standard_least_squares(X_norm, Y_norm)
    plt.figure()
    plt.title('Linear least square method')
    plt.plot(X_norm, Y_norm, 'ro')
    plt.xlabel('Ages')
    plt.ylabel('Insurance Cost')
    plt.plot(X_norm, Y_corrected, 'b-')
    plt.show()
    
    #Plotting Total least square method
    X_corrected, Y_corrected = total_least_squares(X_norm, Y_norm)
    plt.figure()
    plt.title('Total least square method')
    plt.plot(X_norm, Y_norm, 'ro')
    plt.xlabel('Ages')
    plt.ylabel('Insurance Cost')
    plt.plot(X_corrected, Y_corrected, 'b-')
    plt.show()
    
    #Plotting RANSAC method
    threshold = 0.08
    x_line, y_line, x_in, y_in = ransac(X_norm, Y_norm, threshold)
    print("\nThreshold for RANSAC is: ", threshold)
    print("\nBest Fitting Line for RANSAC is through points:", [x_line[0], y_line[0]], [x_line[1], y_line[1]])
    plt.figure()
    plt.title('RANSAC method')
    plt.plot(X_norm, Y_norm, 'ro')
    plt.xlabel('Ages')
    plt.ylabel('Insurance Cost')
    plt.plot(x_in, y_in, 'b*', label = 'inliers')
    plt.plot(x_line, y_line, 'g-', lw = 4, label = 'RANSAC')
    plt.legend()
    plt.show()