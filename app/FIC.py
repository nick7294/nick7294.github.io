import tkinter
import tkinter.messagebox
import customtkinter
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter import filedialog, messagebox
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
import cv2
from PIL import ImageTk, Image 
import os
import time
from skimage.color import rgb2gray
import webbrowser

w=Tk()

def loading():
    width_of_window = 427
    height_of_window = 250
    screen_width = w.winfo_screenwidth()
    screen_height = w.winfo_screenheight()
    x_coordinate = (screen_width/2)-(width_of_window/2)
    y_coordinate = (screen_height/2)-(height_of_window/2)
    w.geometry("%dx%d+%d+%d" %(width_of_window,height_of_window,x_coordinate,y_coordinate))
    w.overrideredirect(1)


    Frame(w, width=427, height=250, bg='#272727').place(x=0,y=0)
    label1=Label(w, text='FIC', fg='green', bg='#272727') 
    label1.configure(font=("Game Of Squids", 24, "bold"))
    label1.place(x=185,y=90)
    label2=Label(w, text='Loading...', fg='green', bg='#272727')
    label2.configure(font=("Calibri", 11))
    label2.place(x=10,y=215)


    image_a=ImageTk.PhotoImage(Image.open(r"images/c2.png"))
    image_b=ImageTk.PhotoImage(Image.open(r"images/c1.png"))




    for i in range(5):
        l1=Label(w, image=image_a, border=0, relief=SUNKEN).place(x=180, y=145)
        l2=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=200, y=145)
        l3=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=220, y=145)
        l4=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=240, y=145)
        w.update_idletasks()
        time.sleep(0.5)

        l1=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=180, y=145)
        l2=Label(w, image=image_a, border=0, relief=SUNKEN).place(x=200, y=145)
        l3=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=220, y=145)
        l4=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=240, y=145)
        w.update_idletasks()
        time.sleep(0.5)

        l1=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=180, y=145)
        l2=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=200, y=145)
        l3=Label(w, image=image_a, border=0, relief=SUNKEN).place(x=220, y=145)
        l4=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=240, y=145)
        w.update_idletasks()
        time.sleep(0.5)

        l1=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=180, y=145)
        l2=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=200, y=145)
        l3=Label(w, image=image_b, border=0, relief=SUNKEN).place(x=220, y=145)
        l4=Label(w, image=image_a, border=0, relief=SUNKEN).place(x=240, y=145)
        w.update_idletasks()
        time.sleep(0.5)


    w.destroy()
    w.mainloop()
    
loading()

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("green")




class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.iconbitmap(r'images/app.ico')
        

        result_text = tkinter.StringVar()

        def select_file():
            file_path = filedialog.askopenfilename()
            keep = float(self.entry.get()) / 100
            process_image(file_path, keep)


        def convert_to_image():
            file_path = filedialog.askopenfilename(filetypes=[("Compressed files", "*.npz")])
            with np.load(file_path) as data:
                Btlow = data['Btlow']
                Alow = np.fft.ifft2(Btlow).real
                plt.figure()
                plt.imshow(Alow, cmap='gray')
                plt.axis('off')
                plt.show()



        def select_image1():
            global image1
            file_path = filedialog.askopenfilename()
            image1 = cv2.imread(file_path)


        def select_image2():
            global image2
            file_path = filedialog.askopenfilename()
            image2 = cv2.imread(file_path)


        def compare_images():
            ssim = compare_ssim(image1, image2, multichannel=True)
            result_text.set(f"SSIM: {ssim}")

        def process_image(file_path, keep):
            A = io.imread(file_path)
            B = np.mean(A, -1)
            dim = A.shape
            print('Original Dimensions:', dim)
            Bt = np.fft.fft2(B)
            Btsort = np.sort(np.abs(Bt.reshape(-1)))
            thresh = Btsort[int(np.floor((1-keep)*len(Btsort)))]
            ind = np.abs(Bt) > thresh
            Btlow = Bt * ind
            save_question = messagebox.askyesno(title="Save the compressed file", message="Do you want to save a compressed file?")
            if save_question:
                file_path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("Compressed files", "*.npz")])
                np.savez_compressed(file_path, Btlow=Btlow)
            Alow = np.fft.ifft2(Btlow).real
            plt.figure()
            plt.imshow(Alow, cmap='gray')
            plt.axis('off')
            plt.show()



        def fourier_masker_hor(image, i):
            f_size = 15
            dark_image_grey = rgb2gray(image)
            dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
            dark_image_grey_fourier[235:240, :230] = i
            dark_image_grey_fourier[235:240,-230:] = i
            fig, ax = plt.subplots(1,3,figsize=(15,15))
            ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
            ax[0].set_title('Masked Fourier', fontsize = f_size)
            ax[1].imshow(dark_image_grey, cmap = 'gray')
            ax[1].set_title('Greyscale Image', fontsize = f_size)
            ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), 
                            cmap='gray')
            ax[2].set_title('Transformed Greyscale Image', 
                            fontsize = f_size)
            plt.show()

        def select_view_image():
            root = Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
            if not file_path:
                print("No file selected.")
                return
            dark_image = plt.imread(file_path)
            fourier_masker_hor(dark_image, 1)
        
        self.geometry(f"{1100}x{580}")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2,), weight=1)
        self.entry = None
        self.string_input_button = None
        self.string_input_button_ssim1 = None
        self.string_input_button_ssim2 = None
        self.string_input_button_ssim3 = None
        self.npy = None
        self.result_text = None
        self.view_button = None    
        
        def appinfo(event):
            webbrowser.open_new("https://nick7294.github.io/appinfo.html")
            
        def kompresinfo(event):
            webbrowser.open_new("https://nick7294.github.io/imgcomp.html")
            
        def ssiminfo(event):
            webbrowser.open_new("https://nick7294.github.io/SSIMcomp.html")
            
        def viewinfo(event):
            webbrowser.open_new("https://nick7294.github.io/scomp.html")
            

        def kompresia():
            if self.string_input_button_ssim1 is not None:
                self.string_input_button_ssim1.grid_forget()
            if self.string_input_button_ssim2 is not None:
                self.string_input_button_ssim2.grid_forget()
            if self.string_input_button_ssim3 is not None:
                self.string_input_button_ssim3.grid_forget()
            if self.result_text is not None:
                self.result_text.grid_forget()
            if self.view_button is not None:
                self.view_button.grid_forget()

            self.title("Image Compression")
            self.textbox_1 = customtkinter.CTkTextbox(self, width=250, font=customtkinter.CTkFont(size=14), wrap = "word")
            self.textbox_1.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
            self.textbox_1.insert("0.0", "Image Compression\n\n" + "This function lets you shrink your images by using a clever math trick called the Fourier transform. This trick changes your image into a bunch of waves with different colors and strengths. You can pick how many waves you want to keep by putting in the % of data you want to keep. The more waves you keep, the better your image looks and the bigger it is. The fewer waves you keep, the worse your image looks and the smaller it is. FIC then turns your image back to normal and lets you save or share it.\n\n" + "For more information visit our website\n\n"  + "WARNING: values have to be numerical and in the following format: [11.4]")
            self.textbox_1.tag_add("hyper", "5.0", "5.end")
            self.textbox_1.tag_config("hyper", foreground="#2CC572", underline=True)
            self.textbox_1.tag_add("bold", "5.0", "5.end")
            self.textbox_1.tag_bind("hyper", "<Button-1>", kompresinfo)
            self.string_input_button = customtkinter.CTkButton(master=self, text="Select an image", command=select_file, width=200)
            self.string_input_button.grid(row=2, column=0, columnspan=4, padx=(200, 20), pady=(20, 20), sticky="n")
            self.entry = customtkinter.CTkEntry(self, placeholder_text="Enter a value from 0.1% to 100%:",width=500)
            self.entry.grid(row=1, column=0, columnspan=2, padx=(200, 0), pady=(20, 0), sticky="n")
            self.npy = customtkinter.CTkButton(master=self,text="Convert to Image", command= convert_to_image, width=200)
            self.npy.grid(row=3, column=0, columnspan=4, padx=(200, 20), pady=(20, 20), sticky="n")
            self.textbox.grid_forget()
            

        def ssim():
            if self.entry is not None:
                self.entry.grid_forget()
            if self.string_input_button is not None:
                self.string_input_button.grid_forget()
            if self.npy is not None:
                self.npy.grid_forget()
            if self.view_button is not None:
                self.view_button.grid_forget()

            self.textbox.grid_forget()
            
            
            self.title("SSIM Calculation")
            self.textbox_2 = customtkinter.CTkTextbox(self, width=250, font=customtkinter.CTkFont(size=14), wrap = "word")
            self.textbox_2.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
            self.textbox_2.insert("0.0", "SSIM Calculation\n\n" + "Fourier Image Compressor also lets you compare your compressed images using a metric called SSIM (Structural Similarity Index Measure). SSIM is a way of measuring how similar two images are based on their structural information. Structural information refers to the patterns of pixels that form edges and shapes in your image. SSIM can help you evaluate how well your compression preserves the important details of your image and how close it is to the original image.\n\n" + "For more information visit our website\n\n" + "WARNING: compared images have to have the same dimensions")
            self.textbox_2.tag_add("hyper", "5.0", "5.end")
            self.textbox_2.tag_config("hyper", foreground="#2CC572", underline=True)
            self.textbox_2.tag_add("bold", "5.0", "5.end")
            self.textbox_2.tag_bind("hyper", "<Button-1>", ssiminfo)
            self.string_input_button_ssim1 = customtkinter.CTkButton(master=self, text="Select the first image", command=select_image1)
            self.string_input_button_ssim1.grid(row=1, column=1, padx=(0, 200), pady=(20, 20), sticky="n")
            self.string_input_button_ssim2 = customtkinter.CTkButton(master=self, text="Select the second image", command=select_image2)
            self.string_input_button_ssim2.grid(row=1, column=1, padx=(200, 0), pady=(20, 20), sticky="n")
            self.string_input_button_ssim3 = customtkinter.CTkButton(master=self, text="Compare", command=compare_images)
            self.string_input_button_ssim3.grid(row=2, column=1, padx=(20, 20), pady=(20, 20), sticky="n")
            self.result_text = customtkinter.CTkLabel(master=self, textvariable=result_text)
            self.result_text.grid(row=3, column=1, padx=(20, 20), pady=(20, 20), sticky="n")
            
        def view():
            if self.entry is not None:
                self.entry.grid_forget()
            if self.string_input_button is not None:
                self.string_input_button.grid_forget()
            if self.string_input_button_ssim1 is not None:
                self.string_input_button_ssim1.grid_forget()
            if self.string_input_button_ssim2 is not None:
                self.string_input_button_ssim2.grid_forget()
            if self.npy is not None:
                self.npy.grid_forget()
            if self.string_input_button_ssim3 is not None:
                self.string_input_button_ssim3.grid_forget()
            if self.result_text is not None:
                self.result_text.grid_forget()
                
                
            self.title("View Compressed")
            self.textbox_3 = customtkinter.CTkTextbox(self, width=250, font=customtkinter.CTkFont(size=14), wrap = "word")
            self.textbox_3.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
            self.textbox_3.insert("0.0", "View Compressed\n\n" + "See your image as waves with Fourier Image Compressor! This app shows you how your image looks in the wave domain. You can learn how the wave domain affects your image quality and size. You can also compare different images as waves and see how they have different wave patterns.\n\n" + "For more information visit our website")
            self.textbox_3.tag_add("hyper", "5.0", "5.end")
            self.textbox_3.tag_config("hyper", foreground="#2CC572", underline=True)
            self.textbox_3.tag_add("bold", "5.0", "5.end")
            self.textbox_3.tag_bind("hyper", "<Button-1>", viewinfo)
            
            self.view_button = customtkinter.CTkButton(master=self, text="Select an image", command=select_view_image)
            self.view_button.grid(row=1, column=0, columnspan=4, padx=(200, 20), pady=(20, 20), sticky="n")
            
        
            


        def info():
            if self.entry is not None:
                self.entry.grid_forget()
            if self.string_input_button is not None:
                self.string_input_button.grid_forget()
            if self.string_input_button_ssim1 is not None:
                self.string_input_button_ssim1.grid_forget()
            if self.string_input_button_ssim2 is not None:
                self.string_input_button_ssim2.grid_forget()
            if self.npy is not None:
                self.npy.grid_forget()
            if self.string_input_button_ssim3 is not None:
                self.string_input_button_ssim3.grid_forget()
            if self.result_text is not None:
                self.result_text.grid_forget()
            if self.view_button is not None:
                self.view_button.grid_forget()
                
            self.title("Home")
            self.textbox = customtkinter.CTkTextbox(self, width=250, font=customtkinter.CTkFont(size=14), wrap="word")
            self.textbox.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
            self.textbox.insert("0.0", "Information about this APP\n\n" + "The Fourier Image Compressor is an app that allows you to compress your images using the Fourier transform. The Fourier transform converts your image into a combination of waves with different frequencies and amplitudes. You can then adjust how many waves you want to retain and see how it affects your image size and quality. You can also view your image in the wave domain and compare it with other images. this app helps you understand the principles of image compression and wave domain.\n\n" + "For more information visit our website")
            self.textbox.tag_add("hyper", "5.0", "5.end")
            self.textbox.tag_config("hyper", foreground="#2CC572", underline=True)
            self.textbox.tag_add("bold", "5.0", "5.end")
            self.textbox.tag_bind("hyper", "<Button-1>", appinfo)


        info()

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="FIC", font=customtkinter.CTkFont(family="Game Of Squids", size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Home", command=info)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Image Compression", command=kompresia)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="SSIM Calculation", command=ssim)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="View Compressed", command=view)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                        command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))
        self.sidebar_button_3.configure()
        self.appearance_mode_optionemenu.set("Dark")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

if __name__ == "__main__":
    app = App()
    app.mainloop()