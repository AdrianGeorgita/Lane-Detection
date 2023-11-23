# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np

def main():

    cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

    screenWidth = 1920
    screenHeight = 1080

    width = int(screenWidth/5)

    while True:
        ret, frame = cam.read()

        if not ret or len(frame) <= 0:
            print("Failed to receive frame")
            break

        # Initializations

        left_top_y = 0
        left_top_x = 0
        left_bottom_y = 0
        left_bottom_x = 0
        right_top_y = 0
        right_top_x = 0
        right_bottom_y = 0
        right_bottom_x = 0

        # 2 Shrink

        ratio = width / frame.shape[1]
        height = int(frame.shape[0] * ratio)

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        cv2.imshow("Original", frame)
        originalFrame = frame.copy()

        # 3 RGB -> Grayscale (Y←0.299⋅R+0.587⋅G+0.114⋅B)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale", frame)

        # 4 Trapez

        trapezoidFrame = np.zeros(frame.shape, dtype = np.uint8)

        upper_left = (int(width * 0.45), int(height * 0.75))
        upper_right = (int(width * 0.55), int(height * 0.75))
        lower_left = (int(width * 0), int(height * 1))
        lower_right = (int(width * 1), int(height * 1))

        trapezoidBounds = np.array([upper_right,lower_right,lower_left,upper_left],dtype = np.int32)

        cv2.fillConvexPoly(trapezoidFrame, trapezoidBounds, 1)

        cv2.imshow("Trapezoid", trapezoidFrame * 255)

        frame = frame * trapezoidFrame

        # 5 Top-Down View

        frameBounds = np.array([(width, 0), (width, height), (0, height), (0, 0)], dtype= np.int32)

        trapezoidBounds = np.float32(trapezoidBounds)
        frameBounds = np.float32(frameBounds)

        magicalMatrix = cv2.getPerspectiveTransform(trapezoidBounds, frameBounds)

        topDownFrame = np.zeros(frame.shape, dtype=np.uint8)
        topDownFrame = cv2.warpPerspective(frame, magicalMatrix, (width, height))
        cv2.imshow("Top-Down", topDownFrame)

        # 6 Blur

        blurredFrame = np.zeros(frame.shape, dtype=np.uint8)
        blurredFrame = cv2.blur(topDownFrame, ksize=(7, 7))
        cv2.imshow("Blurred", blurredFrame)

        # 7 Edge Detection

        sobel_vertical = np.float32([
            [-1, -2, -1],
            [0, 0, 0],
            [+1, +2, +1]
        ])

        sobel_horizontal = np.transpose(sobel_vertical)

        sobelFrame = cv2.filter2D(np.float32(blurredFrame), -1, sobel_vertical)
        sobelFrame2 = cv2.filter2D(np.float32(blurredFrame), -1, sobel_horizontal)

        cv2.imshow("Sobel Frame1", cv2.convertScaleAbs(sobelFrame))
        cv2.imshow("Sobel Frame2", cv2.convertScaleAbs(sobelFrame2))

        sobelFrameFinal = np.sqrt(sobelFrame ** 2 + sobelFrame2 ** 2)
        cv2.imshow("Sobel Frame Final", cv2.convertScaleAbs(sobelFrameFinal))

        # 8 Threshold (Binarize)

        binarizedFrame = cv2.convertScaleAbs(sobelFrameFinal)
        ret, binarizedFrame = cv2.threshold(binarizedFrame, 70, 255, cv2.THRESH_BINARY)
        cv2.imshow("Binarized Image", binarizedFrame)

        # 9 Coordinates of the street markings

        percentage = 0.1
        streetMarkingsFrame = binarizedFrame.copy()
        streetMarkingsFrame[0:height, 0:(int(width * percentage))] = 0
        streetMarkingsFrame[0:height, (int(width * (1 - percentage))):width] = 0
        # cv2.imshow("Street Markings Frame", streetMarkingsFrame)

        leftCoordinates = np.argwhere(streetMarkingsFrame[0:height, 0:(int(width / 2)) - 1] > 0)

        rightCoordinates = np.argwhere(streetMarkingsFrame[0:height, (int(width / 2)):width] > 0)

        if len(leftCoordinates > 0):
            left_ys = leftCoordinates[0:len(leftCoordinates), 0]
            left_xs = leftCoordinates[0:len(leftCoordinates), 1]

        if len(rightCoordinates > 0):
            right_ys = rightCoordinates[0:len(rightCoordinates), 0]
            right_xs = rightCoordinates[0:len(rightCoordinates), 1] + int(width/2)

        # 10 Find the lines

        # y = ax + b,   (b, a)
        leftLine = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
        rightLine = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

        left_top_y = 0
        if -(10 ** 8) <= (-leftLine[0])/leftLine[1] <= 10**8:
            left_top_x = (-leftLine[0])/leftLine[1]

        left_bottom_y = height
        if -(10 ** 8) <= (height - leftLine[0])/leftLine[1] <= 10 ** 8:
            left_bottom_x = (height - leftLine[0])/leftLine[1]

        right_top_y = 0
        if (-10 ** 8) <= (-rightLine[0])/rightLine[1] <= 10 ** 8:
            right_top_x = (-rightLine[0])/rightLine[1]

        right_bottom_y = height
        if (-10 ** 8) <= (height - rightLine[0])/rightLine[1] <= 10 ** 8:
            right_bottom_x = (height - rightLine[0])/rightLine[1]

        left_top = (int(left_top_x), int(left_top_y))
        left_bottom = (int(left_bottom_x), int(left_bottom_y))
        right_top = (int(right_top_x), int(right_top_y))
        right_bottom = (int(right_bottom_x), int(right_bottom_y))

        # print(str(left_top) + " " + str(left_bottom))
        # print(str(right_top) + " " + str(right_bottom))

        cv2.line(streetMarkingsFrame, left_top, left_bottom, (200, 0, 0), 5)
        cv2.line(streetMarkingsFrame, right_top, right_bottom, (100, 0, 0), 5)

        cv2.line(streetMarkingsFrame, (int(width/2), 0), (int(width/2), height), (255, 0, 0), 1)

        cv2.imshow("Street Markings Frame", streetMarkingsFrame)

        # 11 Final Visualization

        leftLineFrame = np.zeros(frame.shape, dtype=np.uint8)
        cv2.line(leftLineFrame, left_top, left_bottom, (255, 0, 0), 5)
        magicalMatrix = cv2.getPerspectiveTransform(frameBounds, trapezoidBounds)
        leftLineFrame = cv2.warpPerspective(leftLineFrame, magicalMatrix, (width, height))
        leftLineCoordinates = np.argwhere(leftLineFrame > 0)
        # left_ys = leftLineCoordinates[0:len(leftLineCoordinates), 0]
        # left_xs = leftLineCoordinates[0:len(leftLineCoordinates), 1]

        rightLineFrame = np.zeros(frame.shape, dtype=np.uint8)
        cv2.line(rightLineFrame, right_top, right_bottom, (255, 0, 0), 5)
        magicalMatrix = cv2.getPerspectiveTransform(frameBounds, trapezoidBounds)
        rightLineFrame = cv2.warpPerspective(rightLineFrame, magicalMatrix, (width, height))
        rightLineCoordinates = np.argwhere(rightLineFrame > 0)
        # right_ys = rightLineCoordinates[0:len(rightLineCoordinates), 0]
        # right_xs = rightLineCoordinates[0:len(rightLineCoordinates), 1]

        # cv2.imshow("Left line", leftLineFrame)
        # cv2.imshow("Right line", rightLineFrame)

        finalFrame = originalFrame.copy()

        for y,x in leftLineCoordinates:
            finalFrame[y][x] = (50, 50, 250)

        for y, x in rightLineCoordinates:
            finalFrame[y][x] = (50, 250, 255)

        cv2.imshow("Final Frame", finalFrame)

        if ret is False:
            break

        cv2.imshow('Trapezoid with Road', frame)
        # cv2.moveWindow('Original', 0, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
