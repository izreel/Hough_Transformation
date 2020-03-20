import cv2

'''
https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#houghcircles

'''

def hough_transform(image, output_name):

    circles = cv2.HoughCircles(image=image, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=200, param2=10, minRadius=10, maxRadius=80)
    output_file = open('opencv_results\output{}.txt'.format(output_name), 'w')

    if circles is not None:
        output_file.write('Number of circles in image: {} \n'.format(circles.shape[1]))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        #taken from example in link in the comments above
        #similar approach used for task 2 implementation
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

            circle_center = 'Center: ({},{})'.format(i[1], i[0])
            circle_radius = 'Radius: {}'.format(i[2])
            output_file.write(circle_center + '\t' + circle_radius)

    cv2.imwrite('opencv_results\output_image{}.jpg'.format(output_name), image)