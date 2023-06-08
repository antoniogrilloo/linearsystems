import os
import tkinter
from threading import Thread
from tkinter import W, TRUE, FALSE
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

import customtkinter
import numpy as np
from customtkinter import StringVar

from methods.IterativeMethod import IterativeMethod
from methods.validation.Validate import Validate


class UserInterface:

    def __init__(self):
        self.open_button = None
        self.method_options = [
            "Jacobi",
            "GaussSeidel",
            "Gradient",
            "Conjugate"
        ]
        self.tolerance_options = [
            1e-4,
            1e-6,
            1e-8,
            1e-10,
        ]
        self.root = customtkinter.CTk()
        self.height = 480
        self.width = 500
        posx = (self.root.winfo_screenwidth() // 2) - (self.width // 2)
        posy = (self.root.winfo_screenheight() // 2) - (self.height // 2)
        self.root.geometry(str(self.width) + "x" + str(self.height) + "+" + str(posx) + "+" + str(posy))
        self.root.title('Linear System Resolver')

        self.clicked = StringVar()
        self.clicked.set(self.method_options[0])

        self.tolerance = StringVar()
        self.tolerance.set(str(self.tolerance_options[0]))

        self.error = StringVar()
        self.times = StringVar()
        self.iter = StringVar()
        self.matrix = StringVar()
        self.file_chosen = StringVar()
        self.file_chosen.set('Chose File')
        self.last_filename = ''

    def startUI(self):
        frame = customtkinter.CTkFrame(master=self.root)
        frame.pack(pady=20, padx=55, fill="both", expand=FALSE)

        customtkinter.CTkLabel(master=frame, text="Choose:").pack(pady=5, padx=10)

        frame2 = customtkinter.CTkFrame(master=frame, fg_color="transparent")
        frame2.pack(fill="both", expand=TRUE, pady=5, padx=100)

        customtkinter.CTkLabel(master=frame2, text="Method:").grid(row=1, column=1, sticky=W)
        customtkinter.CTkOptionMenu(frame2, values=[
            "Jacobi",
            "GaussSeidel",
            "Gradient",
            "Conjugate"
        ],
        hover=False,
        width=85,
        height=22,
        text_color='white',
        fg_color='#555555', variable=self.clicked).grid(row=1, column=2)

        customtkinter.CTkLabel(master=frame2, text="Matrix:").grid(row=2, column=1, sticky=W)
        self.open_button = customtkinter.CTkButton(
            frame2,
            text=self.file_chosen.get(),
            command=self.select_file,
            width=85,
            height=10,
            text_color='white',
            fg_color='#555555',
            hover_color='#555555'

        )

        self.open_button.grid(row=2, column=2)
        customtkinter.CTkLabel(master=frame2, text="Tolerance:").grid(row=3, column=1, sticky=W)
        self.entry = customtkinter.CTkEntry(master=frame2,
                                            placeholder_text="Entry tol...",
                                            width=85,
                                            height=22,
                                            border_width=2,
                                            corner_radius=5)
        self.entry.grid(row=3, column=2, sticky=W)

        frame3 = customtkinter.CTkFrame(master=frame)
        frame3.pack(pady=20, padx=50, fill="both", expand=TRUE)
        customtkinter.CTkLabel(frame3, text="Time: ", font=("calibri", 14, "bold")).pack(pady=0, padx=10)
        customtkinter.CTkLabel(frame3, textvariable=self.times).pack(pady=0, padx=10)
        customtkinter.CTkLabel(frame3, text="Iteration number: ", font=("calibri", 14, "bold")).pack(pady=0, padx=10)
        customtkinter.CTkLabel(frame3, textvariable=self.iter).pack(pady=0, padx=10)
        customtkinter.CTkLabel(frame3, text="Relative Error: ", font=("calibri", 14, "bold")).pack(pady=0, padx=10)
        customtkinter.CTkLabel(frame3, textvariable=self.error).pack(pady=5, padx=10)
        customtkinter.CTkButton(frame, text="Calculate", command=self.threading, width=150).pack(pady=12, padx=10)
        self.progressbar = customtkinter.CTkProgressBar(frame, orientation="horizontal")
        self.progressbar.configure(mode="indeterminate")

        self.root.resizable(False, False)
        self.root.mainloop()


    def select_file(self):
        filetypes = (
            ('mtx files', '*.mtx'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='.',
            filetypes=filetypes)

        if filename != '':
            self.last_filename = filename

        if filename == '' and self.last_filename !='':
            filename = self.last_filename

        self.matrix.set(filename)
        head_tail = os.path.split(filename)
        if head_tail[1] == '':
            self.open_button.configure(text='Chose File')
        else:
            self.open_button.configure(text=head_tail[1])

    def threading(self):
        # Call work function
        t1 = Thread(target=self.calculate)
        t1.start()

    def calculate(self):
        filename = self.matrix.get()
        try:
            tol = float(self.entry.get())
        except:
            tkinter.messagebox.showerror(title=None, message='Tolerance not valid!')
            return

        if filename == '':
            tkinter.messagebox.showerror(title=None, message='File not selected!')
            return
        if tol == 0:
            tkinter.messagebox.showerror(title=None, message='Tolerance not selected!')
            return
        self.error.set('Calculating...')
        self.times.set('Calculating...')
        self.iter.set('Calculating...')
        self.progressbar.pack(pady=12, padx=10)
        self.progressbar.start()
        self.root.update_idletasks()

        try:
            a, n, _ = IterativeMethod.read_matrix(filename)
        except:
            self.error.set('')
            self.times.set('')
            self.iter.set('')
            self.progressbar.stop()
            self.progressbar.pack_forget()
            self.root.update_idletasks()
            tkinter.messagebox.showerror(title=None, message='Runtime error')
            return

        x = np.ones(n)
        b = a @ x
        tmp = str(self.clicked.get())
        try:
            err, it, tf = Validate.validate_method(tmp, a, b, x, tol)
        except:
            tkinter.messagebox.showerror(title=None, message='Runtime error')
            return
        self.error.set(str(err))
        self.times.set(str("%.5f" % tf))
        self.iter.set(str(it))
        self.progressbar.stop()
        self.progressbar.pack_forget()

