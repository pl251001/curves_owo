import cv2 as cv
import numpy as np
import math


def main():
    # img = cv.imread('/Users/local/Downloads/IMG_9912.jpg')
    img = cv.imread('C:/Users/Lucas/Downloads/IMG_0019.jpg')  # 16 19

    height = img.shape[0]
    width = img.shape[1]
    img = cv.resize(img, (2000, 2000))
    pts1 = np.float32([(0, 0), (0, 2000), (2000, 0), (2000, 2000)])

    pts2 = np.float32([(1000, 0), (1000, 2000), (3000, 0), (3000, 2000)])

    # Apply Perspective Transform Algorithm
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    img = cv.warpPerspective(img, matrix, (4000, 4000))

    topl = [1300, 1700]
    topr = [2600, 1700]
    btml = [1300, 1850]
    btmr = [2600, 1850]

    lst = []
    img = detectCurve(img, topl, topr, btml, btmr, lst)
    for x in reversed(range(len(lst))):
        img = cv.warpPerspective(img, lst[x], (4000, 4000))
        cv.namedWindow("image", cv.WINDOW_NORMAL)
        cv.imshow("image", img)
        cv.waitKey(100)
    pts1 = np.float32([(0, 0), (0, 4000), (4000, 0), (4000, 4000)])
    matrix = cv.getPerspectiveTransform(pts2, pts1)
    img = cv.warpPerspective(img, matrix, (4000, 4000))
    img = cv.resize(img, (width, height))
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.imshow('image', img)
    #cv.imwrite('C:/Users/Lucas/Downloads/left.jpg', img)
    cv.waitKey()
    img.release()
    cv.destroyAllWindows()


def detectCurve(img, topl, topr, btml, btmr, lst):
    p1 = [topl[0], topl[1]]
    p2 = [topr[0], topr[1]]
    p3 = [btml[0], btml[1]]
    p4 = [btmr[0], btmr[1]]
    cimg = img[topl[1]:btmr[1], topl[0]:btmr[0]]

    height = cimg.shape[0]

    data = midLine(cimg)
    if data != 0:
        angle = data[0]
        mid = [data[1][0] + topl[0], data[1][1] + topl[1]]

        rotate(topl, angle, mid, height)
        rotate(topr, angle, mid, height)
        rotate(btml, angle, mid, height)
        rotate(btmr, angle, mid, height)

        pts1 = np.float32([topl, topr,
                           btml, btmr])

        pts2 = np.float32([p1, p2, p3, p4])

        # Apply Perspective Transform Algorithm
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        lst.append(cv.getPerspectiveTransform(pts2, pts1))
        img = cv.warpPerspective(img, matrix, (4000, 4000))
        cv.namedWindow("image", cv.WINDOW_NORMAL)
        cv.imshow("image", img)
        cv.namedWindow("image1", cv.WINDOW_NORMAL)
        cv.imshow('image1', cimg)
        cv.waitKey(100)
        img = detectCurve(img, p1, p2, p3, p4, lst)
    return img


def rotate(point, angle, mid, height):
    pt = point.copy()
    pt[0] = pt[0] - mid[0]
    pt[1] = pt[1] - mid[1]

    point[0] = int((pt[0] * math.cos(np.pi / 2 - angle) - pt[1] * math.sin(np.pi / 2 - angle) + mid[0]) + (
            height / math.tan(angle)))
    point[1] = int(pt[0] * math.sin(np.pi / 2 - angle) + pt[1] * math.cos(np.pi / 2 - angle) + mid[1] - height)


def midLine(img):
    if np.shape(img) != ():
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Use canny edge detection
        edges = cv.Canny(gray, 5000, 8000, apertureSize=5)

        kernel_size = 15
        edges = cv.GaussianBlur(edges, (kernel_size, kernel_size), 0)

        lines = cv.HoughLinesP(
            edges,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 180,  # Angle resolution in radians
            threshold=10,  # Min number of votes for valid line
            minLineLength=53,  # Min allowed length of line
            maxLineGap=5  # Max allowed gap between line for joining them
        )
        min = 9999
        left = []
        max = 0
        right = []
        if lines is not None:
            for pt in lines:
                x1, y1, x2, y2 = pt[0]
                if x1 < min:
                    min = x1
                    left = [x1, y1, x2, y2]

                if x2 < min:
                    min = x2
                    left = [x1, y1, x2, y2]
            for pt in lines:
                x1, y1, x2, y2 = pt[0]
                if x1 > max:
                    max = x1
                    right = [x1, y1, x2, y2]

                if x2 > max:
                    max = x2
                    right = [x1, y1, x2, y2]

            if left[1] > left[3]:
                downl = [left[0], left[1]]
                upl = [left[2], left[3]]
            else:
                downl = [left[2], left[3]]
                upl = [left[0], left[1]]

            if right[1] > right[3]:
                downr = [right[0], right[1]]
                upr = [right[2], right[3]]
            else:
                downr = [right[2], right[3]]
                upr = [right[0], right[1]]

            midpt = ((downl[0] + downr[0]) // 2, (downl[1] + downr[1]) // 2)
            endpt = ((upl[0] + upr[0]) // 2, (upl[1] + upr[1]) // 2)
            img = cv.arrowedLine(img, midpt, endpt, (255, 0, 0), 10)
            if (midpt[1] - endpt[1]) / (endpt[0] - midpt[0]) < 0:
                angle = np.pi - math.atan(-(midpt[1] - endpt[1]) / (endpt[0] - midpt[0]))
            else:
                angle = math.atan((midpt[1] - endpt[1]) / (endpt[0] - midpt[0]))

            return [angle, endpt]
        return 0
    else:
        return 0


if __name__ == '__main__':
    main()
