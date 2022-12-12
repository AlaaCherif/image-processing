import math
import numpy as np
import time
from random import randrange
import cv2
import matplotlib.pyplot as plt


class Image:
    matrix = None
    type = ""
    width = 0
    height = 0
    max_gray = 0

    def __init__(self, matrix=None, type="", width=0, height=0, max_gray=0):
        self.matrix = matrix
        self.type = type
        self.width = width
        self.height = height
        self.max_gray = max_gray

    def load_from_pgm(self, pgmPath):

        file = open(pgmPath, 'rb')
        self.type = file.readline().decode().strip()
        line = file.readline()
        while chr(line[0]) == '#':
            line = file.readline()
        width_binary, height_binary = line.split()
        self.width, self.height = int(width_binary), int(height_binary)
        self.max_gray = int(file.readline())
        self.matrix = []
        if(self.type == "P5"):
            for i in range(self.height):
                row = list(file.read(self.width))
                self.matrix.append(row)
        elif(self.type == "P2"):
            for i in range(self.height):
                line = file.readline()
                row = line.split()
                row = list(map(int, row))
                self.matrix.append(row)

        file.close()
    def load_from_ppm(self, ppmPath):

        image=plt.imread(ppmPath)
        self.matrix=image
        self.width=image.shape[0]
        self.height=image.shape[1]
        self.max_gray=255
        # file = open(ppmPath, 'rb')
        # self.type = file.readline().decode().strip()
        # line = file.readline()
        # while chr(line[0]) == '#':
        #     line = file.readline()
        # width_binary, height_binary = line.split()
        # self.width, self.height = int(width_binary), int(height_binary)
        # self.max_gray = int(file.readline())
        # self.matrix = []
        # if(self.type == "P6"):
        #     for i in range(self.height):
        #         row = list(file.read(self.width))
        #         self.matrix.append(row)
        # elif(self.type == "P2"):
        #     for i in range(self.height):
        #         line = file.readline()
        #         row = line.split()
        #         row = list(map(int, row))
        #         self.matrix.append(row)
        # file.close()

    def average(self):
        sum = 0
        for h in range(self.height):
            for w in range(self.width):
                sum += self.matrix[h][w]
        return sum / (self.width*self.height)

    def replace_seuil(self,image, threshold : int) :
        return np.where(image >= threshold , 255  , 0)

    def seuillage_color_blue(self, seuil ) :
        red_channel , green_channel , blue_channel = self.matrix[:,:,2],  self.matrix[:,:,1] , self.matrix[:,:,0]
        image_seuil_blue = self.replace_seuil(blue_channel , seuil)
        image_seuil_red = self.replace_seuil(red_channel , seuil)
        image_seuil_green = self.replace_seuil(green_channel , seuil)
        result = np.dstack((image_seuil_red,image_seuil_green,image_seuil_blue))
        new_image=Image(matrix=result,type="P6",width=self.width,height=self.height,max_gray=self.max_gray)
        return new_image

    def image_seuil_and(self  , r , g , b ) :
        red_channel , green_channel , blue_channel = self.matrix[:,:,2],  self.matrix[:,:,1] , self.matrix[:,:,0]
        image_seuil_red = self.replace_seuil(red_channel , r)
        image_seuil_green = self.replace_seuil(green_channel , g)
        image_seuil_blue = self.replace_seuil(blue_channel , b)
        result = np.bitwise_and(image_seuil_red , image_seuil_green , image_seuil_blue)
        new_image=Image(matrix=result,type="P2",width=self.width,height=self.height,max_gray=self.max_gray)
        return new_image

    def image_seuil_or(self  , r , g , b ) :
        red_channel , green_channel , blue_channel = self.matrix[:,:,2],  self.matrix[:,:,1] , self.matrix[:,:,0]
        image_seuil_red = self.replace_seuil(red_channel , r)
        image_seuil_green = self.replace_seuil(green_channel , g)
        image_seuil_blue = self.replace_seuil(blue_channel , b)
        result = np.bitwise_or(image_seuil_red , image_seuil_green , image_seuil_blue)
        new_image=Image(matrix=result,type="P2",width=self.width,height=self.height,max_gray=self.max_gray)
        return new_image

    def otsu_threshold(self):
        ret2,th2=cv2.threshold(self.matrix,0,255,cv2.THRESH_OTSU)
        new_image=Image(matrix=th2,type="P2",width=self.width,height=self.height,max_gray=self.max_gray)
        return new_image

    def standard_deviation(self):
        average = self.average()
        sum = 0
        for h in range(self.height):
            for w in range(self.width):
                sum += (self.matrix[h][w] - average) ** 2
        return math.sqrt(sum / (self.width*self.height))

    def histogram(self):
        histogram = [0] * (self.max_gray + 1)
        for h in range(self.height):
            for w in range(self.width):
                histogram[self.matrix[h][w]] += 1
        return histogram

    def cumulated_histogram(self):
        histogram = self.histogram()
        cumulated_histogram = [0] * (self.max_gray + 1)
        cumulated_histogram[0] = histogram[0]
        for i in range(1, self.max_gray + 1):
            cumulated_histogram[i] = histogram[i] + cumulated_histogram[i - 1]
        return cumulated_histogram

    def saturation_transformation(self, saturation_points):
        # saturation_points is an array of tuples (x,y)
        trans_x = [0]
        trans_y = [0]

        for point in saturation_points:
            x, y = point
            trans_x.append(x)
            trans_y.append(y)

        trans_x.append(self.max_gray)
        trans_y.append(self.max_gray)

        LUT = [0]*(self.max_gray+1)

        for g in range(self.max_gray+1):
            LUT[g] = int(np.interp(g, trans_x, trans_y))

        new_matrix = []

        for h in range(self.height):
            row = []
            for w in range(self.width):
                row.append(LUT[self.matrix[h][w]])
            new_matrix.append(row)

        new_image = Image(matrix=new_matrix, type="P2", width=self.width,
                          height=self.height, max_gray=self.max_gray)
        return new_image

    def save_to_pgm(self, path="samples/output/{0}.pgm".format(round(time.time()))):

        f = open(path, "w")
        f.write("{0}\n{1} {2}\n{3}\n".format(
            self.type, self.width, self.height, self.max_gray))

        for h in range(self.height):
            f.write(' '.join([str(elem) for elem in self.matrix[h]])+'\n')
        f.close()

    def histogram_equalizer(self):

        cumulated_histogram = self.cumulated_histogram()
        LUT = [0] * (self.max_gray + 1)
        for i in range(self.max_gray + 1):
            LUT[i] = int(self.max_gray * cumulated_histogram[i] /
                         (self.height*self.width))
        new_matrix = []
        for h in range(self.height):
            row = []
            for w in range(self.width):
                row.append(LUT[self.matrix[h][w]])
            new_matrix.append(row)

        new_image = Image(matrix=new_matrix, type="P2", width=self.width,
                          height=self.height, max_gray=self.max_gray)
        return new_image

    def generate_random_noise(self):
        new_matrix = []
        for h in range(self.height):
            row = []
            for w in range(self.width):
                random_val = randrange(21)
                if(random_val == 0):
                    row.append(0)
                elif(random_val == 20):
                    row.append(255)
                else:
                    row.append(self.matrix[h][w])
            new_matrix.append(row)

        new_image = Image(matrix=new_matrix, type="P2", width=self.width,
                          height=self.height, max_gray=self.max_gray)
        return new_image

    def median_filter(self, size):
        if (size % 2 == 0):
            raise Exception('filter size must be odd')

        new_matrix = []
        for h in range(self.height):
            row = []
            for w in range(self.width):
                pixels = []

                for fh in range(h-size//2, h+size//2):
                    for fw in range(w-size//2, w+size//2):
                        if(fh < 0 or fh >= self.height or fw < 0 or fw >= self.width):
                            continue
                        else:
                            pixels.append(self.matrix[fh][fw])

                pixels.sort()
                median = pixels[len(pixels)//2]
                row.append(median)
            new_matrix.append(row)

        new_image = Image(matrix=new_matrix, type="P2", width=self.width,
                          height=self.height, max_gray=self.max_gray)
        return new_image

    def average_filter(self, size):
        if (size % 2 == 0):
            raise Exception('filter size must be odd')

        new_matrix = []
        for h in range(self.height):
            row = []
            for w in range(self.width):
                pixels = []

                for fh in range(h-size//2, h+size//2):
                    for fw in range(w-size//2, w+size//2):
                        if(fh < 0 or fh >= self.height or fw < 0 or fw >= self.width):
                            continue
                        else:
                            pixels.append(self.matrix[fh][fw])
                average = sum(pixels)/len(pixels)
                row.append(int(average))
            new_matrix.append(row)

        new_image = Image(matrix=new_matrix, type="P2", width=self.width,
                          height=self.height, max_gray=self.max_gray)
        return new_image

    @staticmethod
    def signal_to_noise_ratio(original_image, filtered_image):

        orig_img_avg = original_image.average()
        S, B = (0, 0)

        for h in range(original_image.height):
            for w in range(original_image.width):
                S += (original_image.matrix[h][w]-orig_img_avg)**2
                B += (filtered_image.matrix[h][w] -
                      original_image.matrix[h][w])**2

        return math.sqrt(S/B)

    def binarize(self, threshold):

        new_matrix = []
        for h in range(self.height):
            row = []
            for w in range(self.width):
                if self.matrix[h][w] < threshold:
                    row.append(0)
                else:
                    row.append(self.max_gray)
            new_matrix.append(row)

        return Image(matrix=new_matrix, type="P2", width=self.width,
                     height=self.height, max_gray=self.max_gray)

    def dilation(self, size):
        if (size % 2 == 0):
            raise Exception('size must be odd')

        new_matrix = []
        for h in range(self.height):
            row = []
            for w in range(self.width):
                row.append(self.max_gray)
            new_matrix.append(row)

        for h in range(self.height):
            for w in range(self.width):
                if(self.matrix[h][w] == self.max_gray):
                    continue
                else:
                    for dh in range(h-size//2, h+size//2+1):
                        for dw in range(w-size//2, w+size//2+1):
                            if(dh < 0 or dh >= self.height or dw < 0 or dw >= self.width):
                                continue
                            else:
                                new_matrix[dh][dw] = 0

        return Image(matrix=new_matrix, type="P2", width=self.width,
                     height=self.height, max_gray=self.max_gray)
        
    
    def erosion(self,size):
        if (size % 2 == 0):
            raise Exception('size must be odd')
        
        
        new_matrix = []
        for h in range(self.height):
            row = []
            for w in range(self.width):
                row.append(self.matrix[h][w])
            new_matrix.append(row)
            
        
        for h in range(self.height):
            for w in range(self.width):
                
                if(self.matrix[h][w] == self.max_gray):
                    continue
                else:
                    for dh in range(h-size//2, h+size//2+1):
                        for dw in range(w-size//2, w+size//2+1):
                            if(dh < 0 or dh >= self.height or dw < 0 or dw >= self.width):
                                continue
                            else:
                                if(self.matrix[dh][dw]!=0):
                                    new_matrix[h][w]=self.max_gray
        return Image(matrix=new_matrix, type="P2", width=self.width,
                     height=self.height, max_gray=self.max_gray)