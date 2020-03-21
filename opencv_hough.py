import cv2
import matplotlib.pyplot as plt


'''
https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#houghcircles

'''

def hough_transform(image):
    input_image = cv2.imread(image, 0)
    circles = cv2.HoughCircles(image=input_image, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=200, param2=10, minRadius=10, maxRadius=80)

    if circles is not None:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        
        #taken from example in link in the comments above
        #similar approach used for task 2 implementation
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(input_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(input_image, (i[0], i[1]), 2, (0, 0, 255), 3)

            circle_center = 'Center: ({},{})'.format(i[1], i[0])
            circle_radius = 'Radius: {}'.format(i[2])
            print(circle_center, circle_radius)
    
    plt.imshow(input_image)
    plt.show()