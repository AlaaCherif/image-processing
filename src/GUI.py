import tkinter as tk
from tkinter import Menu, filedialog, Canvas, PhotoImage, NW, Label, Text, Entry, simpledialog
from Image import Image
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2


class GUI:
    root = tk.Tk()

    def __init__(self):
        self.root.title('Photoshock')
        self.root.geometry("1280x720")
        self.root.configure(bg="#D2EAD9")
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = Menu(menubar)
        
        file_menu.add_command(
            label='Open PGM',
            command=self.openInput
        )
        file_menu.add_command(
            label='Open PPM',
            command=self.openInputppm
        )

        file_menu.add_command(
            label='Save As',
            command=self.saveOutput
        )

        menubar.add_cascade(
            label="File",
            menu=file_menu
        )

        operations_menu = Menu(menubar)

        operations_menu.add_command(
            label='Equalize',
            command=self.histogram_equalizer
        )
        operations_menu.add_command(
            label='Saturation',
            command=self.saturation_transformation
        )

        operations_menu.add_command(
            label='Make Noise',
            command=self.generate_random_noise
        )

        operations_menu.add_command(
            label='Median Filter',
            command=self.median_filter
        )

        operations_menu.add_command(
            label='Average Filter',
            command=self.average_filter
        )
        operations_menu.add_command(
            label='Threshold',
            command=self.binarize
        )
        operations_menu.add_command(
            label='Dilation',
            command=self.dilation
        )
        operations_menu.add_command(
            label='Erosion',
            command=self.erosion
        )
        
        operations_menu.add_command(
            label='Opening',
            command=self.opening
        )
        
        operations_menu.add_command(
            label='Closing',
            command=self.closing
        )
        menubar.add_cascade(
            label="PGM Operations",
            menu=operations_menu
        )
        ppm_operations = Menu(menubar)
        ppm_operations.add_command(
            label="Threshold",
            command=self.threshold_colors
        )
        ppm_operations.add_command(
            label="And Threshold",
            command=self.and_threshhold
        )
        ppm_operations.add_command(
            label="Or Threshold",
            command=self.or_threshhold
        )
        ppm_operations.add_command(
            label="Otsu Threshold",
            command=self.otsu_threshold
        )


        menubar.add_cascade(
            label="PPM Operations",
            menu=ppm_operations
        )

        flip_menu=Menu(menubar)
        flip_menu.add_command(label="Make Input PGM",command=self.make_input)
        flip_menu.add_command(label="Make Input PPM",command=self.make_input_ppm)
        
        menubar.add_cascade(label="Convert",menu=flip_menu)

        self.initInputCanvas()

        self.initOutputCanvas()
        self.root.mainloop()

    def initInputCanvas(self):

        self.inputCanvas = Canvas(self.root, width=500, height=400)
        Label(self.inputCanvas, text="INPUT").grid(column=1, row=1, sticky='w')
        self.inputCanvas.grid(column=1, row=1, sticky='w')
        self.inputInfo = Canvas(self.inputCanvas)

    def initOutputCanvas(self):

        self.outputCanvas = Canvas(self.root, width=500, height=400)
        Label(self.outputCanvas, text="OUTPUT").grid(
            column=1, row=1, sticky='w')
        self.outputCanvas.grid(column=1, row=2, sticky='w')
        self.outputInfo = Canvas(self.outputCanvas)

    def openInput(self):
        filename = filedialog.askopenfilename(
            initialdir="./samples", title="Select a File")
        self.inputImage = Image()
        self.inputImage.load_from_pgm(filename)
        self.inputCanvas.destroy()
        self.initInputCanvas()
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.inputImage.matrix, cmap='gray', vmin=0, vmax=self.inputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.inputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()

        self.updateInfo('input')
        self.root.mainloop()

    def openInputppm(self):
        filename = filedialog.askopenfilename(
            initialdir="./samples", title="Select a File")
        self.inputImage = Image()
        self.inputImage.load_from_ppm(filename)
        self.inputCanvas.destroy()
        self.initInputCanvas()
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.inputImage.matrix,  vmin=0, vmax=self.inputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.inputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()
        self.updateInfoPpm('input')
        self.root.mainloop()

    def updateInfo(self, element):
        '''element is string can be 'input' or 'output'  '''
        if(element == 'input'):
            canvas = self.inputCanvas
            image = self.inputImage
            infoCanvas = self.inputInfo
        elif(element == 'output'):
            canvas = self.outputCanvas
            image = self.outputImage
            infoCanvas = self.outputInfo
        else:
            return

        infoCanvas.destroy()
        infoCanvas = Canvas(canvas)
        infoCanvas.grid(column=2, row=1, rowspan=2, sticky='n')
        Label(infoCanvas, text=f'height {image.height}').grid(
            column=1, row=1, sticky='w')
        Label(infoCanvas, text=f'width {image.width}').grid(
            column=1, row=2, sticky='w')
        Label(infoCanvas, text=f'average {image.average()}').grid(
            column=1, row=3, sticky='w')
        Label(infoCanvas, text=f'standard_deviation {image.standard_deviation()}').grid(
            column=1, row=4, sticky='w')
        if(element == "output"):
            Label(infoCanvas, text=f'signal to noise ratio {Image.signal_to_noise_ratio(self.inputImage,self.outputImage)}').grid(
                column=1, row=5, sticky='w')

        data = {
            'level': range(image.max_gray+1),
            'nb_pixels': image.histogram()
        }

        df = DataFrame(data, columns=['level', 'nb_pixels'])

        figure = plt.Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        line = FigureCanvasTkAgg(figure, infoCanvas)
        line.get_tk_widget().grid(column=2, row=1, rowspan=4, sticky='w')
        df = df[['level', 'nb_pixels']].groupby('level').sum()
        df.plot(kind='line', legend=True, ax=ax, color='r', fontsize=10)
        ax.set_title('histogram')

        data = {
            'level': range(image.max_gray+1),
            'nb_pixels': image.cumulated_histogram(),
        }

        df = DataFrame(data, columns=['level', 'nb_pixels'])

        figure = plt.Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        line = FigureCanvasTkAgg(figure, infoCanvas)
        line.get_tk_widget().grid(column=3, row=1, rowspan=4, sticky='w')
        df = df[['level', 'nb_pixels']].groupby('level').sum()
        df.plot(kind='line', legend=True, ax=ax, color='r', fontsize=10)
        ax.set_title('cummulated histogram')


    def updateInfoPpm(self, element):
        '''element is string can be 'input' or 'output'  '''
        if(element == 'input'):
            canvas = self.inputCanvas
            image = self.inputImage
            infoCanvas = self.inputInfo
        elif(element == 'output'):
            canvas = self.outputCanvas
            image = self.outputImage
            infoCanvas = self.outputInfo
        else:
            return

        infoCanvas.destroy()
        infoCanvas = Canvas(canvas)
        infoCanvas.grid(column=2, row=1, rowspan=2, sticky='n')
        Label(infoCanvas, text=f'height {image.height}').grid(
            column=1, row=1, sticky='w')
        Label(infoCanvas, text=f'width {image.width}').grid(
            column=1, row=2, sticky='w')
        # Label(infoCanvas, text=f'average {image.average()}').grid(
        #     column=1, row=3, sticky='w')
        # Label(infoCanvas, text=f'standard_deviation {image.standard_deviation()}').grid(
        #     column=1, row=4, sticky='w')
        # if(element == "output"):
        #     Label(infoCanvas, text=f'signal to noise ratio {Image.signal_to_noise_ratio(self.inputImage,self.outputImage)}').grid(
        #         column=1, row=5, sticky='w')
        data = {
            'level': range(256),
            'nb_pixelsr':  cv2.calcHist([image.matrix.astype(np.float32)],[0],None,[256],[0,256]).ravel(),
            'nb_pixelsg':  cv2.calcHist([image.matrix.astype(np.float32)],[1],None,[256],[0,256]).ravel(),
            'nb_pixelsb':  cv2.calcHist([image.matrix.astype(np.float32)],[2],None,[256],[0,256]).ravel()
            
        }

        df = DataFrame(data, columns=['level', 'nb_pixelsr','nb_pixelsg','nb_pixelsb'])

        
        figure = plt.Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        line = FigureCanvasTkAgg(figure, infoCanvas)
        line.get_tk_widget().grid(column=2, row=1, rowspan=4, sticky='w')
        df = df[['level', 'nb_pixelsr','nb_pixelsg','nb_pixelsb']].groupby('level').sum()
        df.plot(kind='line', legend=True, ax=ax, color='rgb', fontsize=10)
        ax.set_title('histogram')

        data = {
            'level': range(256),
            'nb_pixelsr':  cv2.calcHist([image.matrix.astype(np.float32)],[0],None,[256],[0,256]).ravel().cumsum(),
            'nb_pixelsg':  cv2.calcHist([image.matrix.astype(np.float32)],[1],None,[256],[0,256]).ravel().cumsum(),
            'nb_pixelsb':  cv2.calcHist([image.matrix.astype(np.float32)],[2],None,[256],[0,256]).ravel().cumsum()
        }

        df = DataFrame(data, columns=['level', 'nb_pixelsr','nb_pixelsg','nb_pixelsb'])

        figure = plt.Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        line = FigureCanvasTkAgg(figure, infoCanvas)
        line.get_tk_widget().grid(column=3, row=1, rowspan=4, sticky='w')
        df = df[['level', 'nb_pixelsr','nb_pixelsg','nb_pixelsb']].groupby('level').sum()
        df.plot(kind='line', legend=True, ax=ax, color='rgb', fontsize=10)
        ax.set_title('cummulated histogram')

    def histogram_equalizer(self):
        self.outputImage = self.inputImage.histogram_equalizer()
        self.updateOutput()

    def updateOutput(self):

        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.outputImage.matrix, cmap='gray', vmin=0, vmax=self.outputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.outputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()

        self.updateInfo('output')
        self.root.mainloop()

    def saturation_transformation(self):

        answer = simpledialog.askstring(
            "Input",
            "provide saturation points in this form: x1 y1 x2 y2 x3 y3 ...",
            parent=self.root)
        answer = answer.split()
        saturation_points = []
        while(len(answer) != 0):
            saturation_points.append((int(answer.pop(0)), int(answer.pop(0))))

        self.outputImage = self.inputImage.saturation_transformation(
            saturation_points)
        self.updateOutput()
        return

    def generate_random_noise(self):
        self.outputImage = self.inputImage.generate_random_noise()
        self.updateOutput()

    def median_filter(self):

        answer = simpledialog.askstring(
            "Input",
            "provide the median filter size",
            parent=self.root)
        self.outputImage = self.inputImage.median_filter(int(answer))
        self.updateOutput()

    def average_filter(self):
        answer = simpledialog.askstring(
            "Input",
            "provide the average filter size",
            parent=self.root)
        self.outputImage = self.inputImage.average_filter(int(answer))
        self.updateOutput()

    def saveOutput(self):
        
        filename = filedialog.asksaveasfilename(
            initialdir="./samples", title="choose file location")
        self.outputImage.save_to_pgm(filename)
        return
    
    def binarize(self):
        threshold = simpledialog.askstring(
            "Input",
            "provide the threshold",
            parent=self.root)
        self.outputImage=self.inputImage.binarize(int(threshold))
        self.updateOutput()
        
    def dilation(self):
        
        size = simpledialog.askstring(
            "Input",
            "provide the dilation size",
            parent=self.root)
        self.outputImage=self.inputImage.dilation(int(size))
        self.updateOutput()
        
    def erosion(self):
        
        size = simpledialog.askstring(
            "Input",
            "provide the erosion size",
            parent=self.root)
        self.outputImage=Image(matrix=cv2.erode(self.inputImage, np.ones((size, size), np.uint8),iterations=1),type="P2",width=self.inputImage.width,height=self.inputImage.height,max_gray=255)
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.outputImage.matrix, cmap="gray", vmin=0, vmax=self.outputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.outputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()
        self.updateInfo('output')
        self.root.mainloop()
        
    def closing(self):
        size = simpledialog.askstring(
            "Input",
            "provide the closing size",
            parent=self.root)
        self.outputImage=self.inputImage.dilation(int(size)).erosion(int(size))
        self.updateOutput()
        
        
    def opening(self):
        size = simpledialog.askstring(
            "Input",
            "provide the opening size",
            parent=self.root)
        self.outputImage=self.inputImage.erosion(int(size)).dilation(int(size))
        self.updateOutput()
        
    def make_input(self):
        self.inputImage=self.outputImage
        self.inputCanvas.destroy()
        self.initInputCanvas()
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.inputImage.matrix, cmap='gray', vmin=0, vmax=self.inputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.inputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()

        self.updateInfo('input')
        self.root.mainloop()
    
    def make_input_ppm(self):
        self.inputImage=self.outputImage
        self.inputCanvas.destroy()
        self.initInputCanvas()
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.inputImage.matrix, vmin=0, vmax=self.inputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.inputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()

        self.updateInfoPpm('input')
        self.root.mainloop()

    def and_threshhold(self):
        red=simpledialog.askstring(
            "Input",
            "provide red threshold",
            parent=self.root)
        green=simpledialog.askstring(
            "Input",
            "provide green threshold",
            parent=self.root)
        blue=simpledialog.askstring(
            "Input",
            "provide blue threshold",
            parent=self.root)
        threshold_image=self.inputImage.image_seuil_and(int(red),int(green),int(blue))
        self.outputImage=threshold_image
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.outputImage.matrix, cmap="gray", vmin=0, vmax=self.outputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.outputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()
        self.updateInfo('output')
        self.root.mainloop()

    def threshold_colors(self):
        answer=simpledialog.askstring(
            "Input",
            "provide a threshold",
            parent=self.root)
        threshold_image=self.inputImage.seuillage_color_blue(int(answer))
        self.outputImage=threshold_image
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.outputImage.matrix, vmin=0, vmax=self.outputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.outputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()
        self.updateInfoPpm('output')
        self.root.mainloop()

    def and_threshhold(self):
        red=simpledialog.askstring(
            "Input",
            "provide red threshold",
            parent=self.root)
        green=simpledialog.askstring(
            "Input",
            "provide green threshold",
            parent=self.root)
        blue=simpledialog.askstring(
            "Input",
            "provide blue threshold",
            parent=self.root)
        threshold_image=self.inputImage.image_seuil_and(int(red),int(green),int(blue))
        self.outputImage=threshold_image
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.outputImage.matrix, cmap="gray", vmin=0, vmax=self.outputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.outputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()
        self.updateInfo('output')
        self.root.mainloop()

    def or_threshhold(self):
        red=simpledialog.askstring(
            "Input",
            "provide red threshold",
            parent=self.root)
        green=simpledialog.askstring(
            "Input",
            "provide green threshold",
            parent=self.root)
        blue=simpledialog.askstring(
            "Input",
            "provide blue threshold",
            parent=self.root)
        threshold_image=self.inputImage.image_seuil_or(int(red),int(green),int(blue))
        self.outputImage=threshold_image
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.outputImage.matrix, cmap="gray", vmin=0, vmax=self.outputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.outputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()
        self.updateInfo('output')
        self.root.mainloop()

    def otsu_threshold(self):
        threshold_image=self.inputImage.otsu_threshold()
        self.outputImage=threshold_image
        self.outputCanvas.destroy()
        self.initOutputCanvas()
        fig = plt.Figure(figsize=(4, 4), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(self.outputImage.matrix, cmap="gray", vmin=0, vmax=self.outputImage.max_gray)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, self.outputCanvas)
        canvas.get_tk_widget().grid(column=1, row=2, sticky='w')
        canvas.draw()
        self.updateInfo('output')
        self.root.mainloop()