import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from without_hint import without_hint

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.create_buttons()
        self.fullscreen = False
        self.master.bind("<F11>", self.toggle_fullscreen)
        self.master.bind("<Escape>", self.end_fullscreen)

    def create_buttons(self):
        button_frame = tk.Frame(self)
        button_frame.pack(side="top", fill="x")

        insert_one_image_button = tk.Button(button_frame, text="Insert one image", command=self.insert_one_image)
        insert_one_image_button.pack(side="left")

        exit_button = tk.Button(button_frame, text="Exit", fg="red", command=self.master.destroy)
        exit_button.pack(side="right")

    def insert_one_image(self):
        self.insert_image('Puzzle Image')
        self.show_result_image()

    def insert_image(self, title):
        file_path = filedialog.askopenfilename()
        
        try:
            img = Image.open(file_path)
            without_hint(file_path)
            
            
        except IOError:
            print("Invalid image file")
            return

        img = self.resize_image(img)

        frame = tk.Frame(root)
        frame.pack(side="left", fill="both", expand="yes")

        label = tk.Label(frame, text=title)
        label.pack(side="top", fill="both", expand="yes")

        panel = tk.Label(frame, image=img)
        panel.image = img
        panel.pack(side="top", fill="both", expand="yes")

    def show_result_image(self):
        # replace the result image from the program
        result_image_path = 'output.jpg'
        
        try:
            img = Image.open(result_image_path)
        except IOError:
            print("Invalid image file")
            return

        img = self.resize_image(img)

        frame = tk.Frame(root)
        frame.pack(side="right", fill="both", expand="yes")

        label = tk.Label(frame, text='Result Image')
        label.pack(side="top", fill="both", expand="yes")

        panel = tk.Label(frame, image=img)
        panel.image = img
        panel.pack(side="top", fill="both", expand="yes")

    def resize_image(self, img):
        max_size = (self.master.winfo_screenwidth()//3, self.master.winfo_screenheight())
        img.thumbnail(max_size)
        return ImageTk.PhotoImage(img)

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.master.attributes("-fullscreen", self.fullscreen)
        return "break"

    def end_fullscreen(self, event=None):
        self.fullscreen = False
        self.master.attributes("-fullscreen", False)
        return "break"

root = tk.Tk()
app = Application(master=root)
app.mainloop()
