import cv2
import numpy as np
'''
https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#houghcircles

'''

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-i", "--image", dest="image",
                        help="specify the name of the image", metavar="IMAGE")
    args = parser.parse_args()
    input_image = cv2.imread(args.image, 0)

    if '.jpg' in args.image:
        part_output_name = args.image[len(args.image) - 5]
        #built in hough transform from OpenCV
        builtin_transform(input_image, part_output_name)
        #self implemented hough trasformation
        hough_transformation(input_image, part_output_name)


def builtin_transform(image, output_name):

    circles = cv2.HoughCircles(image=image, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=200, param2=10, minRadius=10, maxRadius=80)
    output_file = open('task1\output' + output_name + '.txt', 'w')

    if circles is not None:
        output_file.write("Number of circles in image: " + str(circles.shape[1]) + '\n')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #taken from example in link in the comments above
        #similar approach used for task 2 implementation
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
            output_file.write('Center: (' + str(i[1]) + ',' + str(i[0]) + ')' + ' Radius: ' + str(i[2]) + '\n')

    cv2.imwrite('task1\output_image' +  output_name + '.jpg', image)


def distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def hough_transformation(image, output_name):
    edges = cv2.Canny(image, 10, 200)
    edge_points = np.nonzero(edges)

    if len(edge_points[0] > 0):
        H = np.zeros((image.shape[0], image.shape[1], 30))
        for i in range(len(edge_points[0])):
            for j in range(10, 30):
                for k in range(360):
                    a = int(edge_points[0][i] - j * np.cos(np.radians(k)))
                    b = int(edge_points[1][i] + j * np.sin(np.radians(k)))
                    H[a, b, j] += 1

        circles = []
        voting = []
        for r in range(H.shape[2]):
            max_y = 0
            max_x = 0
            max_votes = 0
            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    if H[i, j, r] > max_votes:
                        max_y = j
                        max_x = i
                        max_votes = H[i, j, r]

            if max_votes > 0:
                voting.append(max_votes)
                circles.append([max_x, max_y, r])

        mean_score = np.mean(voting)
        unique_circles = []
        for i in range(len(voting)):
            if voting[i] >= mean_score:
                unique_circles.append([circles[i][0], circles[i][1], circles[i][2], voting[i]])

        sorted_circles = sorted(unique_circles, key=lambda x: x[3], reverse=True)
        unique_circles = [[], [], []]
        for i in range(len(sorted_circles)-1):
            if sorted_circles[i][0] not in unique_circles[0] and sorted_circles[i][1] not in unique_circles[1]:
                unique_circles[0].append(sorted_circles[i][0])
                unique_circles[1].append(sorted_circles[i][1])
                unique_circles[2].append(sorted_circles[i][2])

        output_file = open('task2\output'+ output_name + '.txt', 'w')
        output_file.write("Number of circles in image: " + str(len(unique_circles[1])) + '\n')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for i in range(len(unique_circles)):
            # draw the outer circle
            cv2.circle(image, (unique_circles[1][i], unique_circles[0][i]), unique_circles[2][i], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (unique_circles[1][i], unique_circles[0][i]), 2, (0, 0, 255), 3)
            output_file.write('Center: (' + str(unique_circles[0][i]) + ',' + str(unique_circles[1][i]) + ')' + ' Radius: ' + str(unique_circles[2][i]) + '\n')

        #creating images for accumulator arrays
        for i in range(len(unique_circles)):

            cv2.imwrite('task2\output_accumulator_' + output_name + '_' + str(unique_circles[2][i]) + '.jpg', H[:, :, unique_circles[2][i]])
    cv2.imwrite('task2\output'+ output_name + '.jpg', image)


main()