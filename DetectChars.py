# DetectChars.py
import os
import cv2
import numpy as np
import math
import pytesseract
from pytesseract import image_to_string

import PossibleChar

# constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 27

MAX_PIXEL_WIDTH = 26
MAX_PIXEL_HEIGHT = 50

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 90

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 20

MIN_CONTOUR_AREA = 100

###################################################################################################

def findPossibleCharsInPlate(imgThresh):
    listOfMatchingChars = [] # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

    # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    character_dimensions = (0.4*imgThreshCopy.shape[0], 0.85*imgThreshCopy.shape[0], 0.04*imgThreshCopy.shape[1], 0.15*imgThreshCopy.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions
    
    for contour in contours: # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar,min_height, max_height, min_width, max_width):
            # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfMatchingChars.append(possibleChar) # add to list of possible chars

    return listOfMatchingChars
# end function

###################################################################################################

def checkIfPossibleChar(possibleChar,min_height, max_height, min_width, max_width):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > min_width and possibleChar.intBoundingRectHeight > min_height and
        possibleChar.intBoundingRectWidth < max_width and possibleChar.intBoundingRectHeight < max_height):
        return True
    else:
        return False
# end function

###################################################################################################

# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################

# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = listOfMatchingChars # this will be the return value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar: # if current char and other char are not the same char . . .
                # if current char and other char have center points at almost the same location . . .
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # if we get in here we have found overlapping chars
                    # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:  # if current char is smaller than other char
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved: # if current char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar) # then remove current char
                        # end if
                    else: # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved: # if other char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar) # then remove other char

    return listOfMatchingCharsWithInnerCharRemoved
# end function

###################################################################################################

# this is where we apply the actual char recognition
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = "" # this will be the return value, the chars in the lic plate

    height, width = imgThresh.shape
    
    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX) # sort chars from left to right

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor) # make color version of threshold image so we can draw contours in color on it
    
    characters = []
    column_list = []
    config = ('-1 eng --oem 1 --psm3')
    strChars = pytesseract.image_to_string(imgThresh, config=config)
    #strChars = pytesseract.image_to_string(imgThresh, lang='eng')
    for currentChar in listOfMatchingChars: # for each char in plate
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))
        cv2.rectangle(imgThreshColor, pt1, pt2, (0, 255, 0), 2) # draw green box around the char
        cv2.imshow('Character Detected',imgThreshColor)
    # end for    
    return strChars
# end function

###################################################################################################

