import os
from tkinter import Tk, Label, Button, Entry, IntVar, END, W, E, messagebox, Toplevel, Frame
from tkinter.filedialog import askdirectory
from preProcessing import preProcessing
from NaiveBayes import naiveBayes
import glob
import pandas as pd

class GUI:

    def __init__(self, master):
        self.data = []
        self.master = master
        master.title("Navie Bayes Classifier")
        master.geometry("500x350+10+20")

        self.path = ""
        self.bin = ""
        self.classifier = False

        self.label = Label(master, text="Directory Path:",bd=5,fg='blue')
        vcmd = master.register(self.validate)  # we have to wrap the command
        self.path = Entry(master)
        self.Browse = Button(master, text="Browse", command=lambda: self.openFileChooser())
        self.Dis = Label(master, text="Discretization Bins:",bd=5,fg='blue')
        self.bin = Entry(master, validate="key", validatecommand=(vcmd, '%P'))
        self.Build = Button(master, text="Build", command=lambda: self.build_model())
        self.Classify = Button(master, text="Classify", command=lambda: self.classify())

        # LAYOUT

        self.label.grid(row=5, column=2, sticky=W)
        self.path.grid(row=5, column=5, columnspan=10, sticky=W+E)
        self.Browse.grid(row=5, column=40)
        self.Dis.grid(row=8, column=2, sticky=W)
        self.bin.grid(row=8, column=5, columnspan=10, sticky=W+E)

        self.Build.grid(row=20, column=6,sticky=W+E)
        self.Classify.grid(row=25, column=6, sticky=W+E)


    def validate(self, new_text):
        if not new_text:  # the field is being cleared
            self.entered_number = 0
            return True

        try:
            self.entered_number = int(new_text)
            return True
        except ValueError:
            return False

    def openFileChooser(self):
        filename = askdirectory()
        self.path.insert(0, filename)

    #pipeline of build the file train
    def build_model(self):
        dir_path = self.path.get()

        if dir_path != "" and self.bin.get() != "":
            bins = int(self.bin.get())
            train_exists = False
            test_exists = False
            structure_exists = False
            train_exists = os.path.isfile("./train.csv")
            test_exists = os.path.isfile("./test.csv")
            structure_exists = os.path.isfile("./structure.txt")

            if train_exists and test_exists and structure_exists:
                try:
                    dir_path = self.path.get()
                    train_empty=pd.read_csv( './train.csv', header=0)
                    test_empty = pd.read_csv('./test.csv', header=0)
                    structure = './structure.txt'
                    file = open(structure, 'r')
                    content = file.readlines()
                    if len(content) != 0:
                        pre = preProcessing(dir_path, bins)
                        pre.prepros()
                        self.data = pre
                        self.classifier = True
                        messagebox.showinfo(title=None, message="Building classifier using train-set is done!")
                    else:
                        messagebox.showinfo(title=None, message="The files is empty!")
                except:
                    messagebox.showinfo(title=None, message="The files is empty!")
            else:
                messagebox.showinfo(title=None, message="The files not exists!")
        else:
            messagebox.showinfo(title=None, message="invalid input!")


    #classify of naive
    def classify(self):
        if self.classifier:
            nb = naiveBayes()
            nb.classify(self.data.df_csv, self.data.df_csv_test, self.data.classess, self.data.attr, self.data.path +'/output.txt')
            dialog = messagebox.showinfo(title="classifiy done", message=" classifier is done!")
            self.master.destroy()
        else:
            dialog = messagebox.showinfo(title="classifiy done", message="please build the model!")


root = Tk()
my_gui = GUI(root)
root.mainloop()
