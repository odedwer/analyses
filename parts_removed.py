### this script will contain parts I removed from the original pipeline


##
# raw.filter(h_freq=None, l_freq=1)


# %% in case of existing raw file, like detrended:
# raw = mne.io.read_raw_fif(input("Hello!\nEnter raw data file: "))
# raw.load_data()
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler

plt.specgram

# Seperated out config of plot to just do it once
def config_plot():
    fig, ax = plt.subplots()
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='Graph One')
    return (fig, ax)


class matplotlibSwitchGraphs:
    def __init__(self, master):
        self.master = master
        self.frame = Frame(self.master)
        self.fig, self.ax = config_plot()
        self.graphIndex = 0
        self.maxIndex = 7
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.config_window()
        self.draw_graph(self.graphIndex)
        self.frame.pack(expand=YES, fill=BOTH)

    def config_window(self):
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.button = Button(self.master, text="Quit", command=self._quit)
        self.button.pack(side=BOTTOM)
        self.button_switch_up = Button(self.master, text="->", command=self.move_up)
        self.button_switch_up.pack(side=RIGHT)
        self.button_switch_down = Button(self.master, text="<-", command=self.move_down)
        self.button_switch_down.pack(side=LEFT)
    def draw_graph(self, index):
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(index * np.pi * t)
        self.ax.clear()  # clear current axes
        self.ax.plot(t, s)
        self.ax.set(title="component " + str(index))
        self.canvas.draw()
    def on_key_press(event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, toolbar)
    def _quit(self):
        self.master.quit()  # stops mainloop
    def move_up(self):
        # Need to call the correct draw, whether we're on graph one or two
        self.graphIndex = (self.graphIndex + 1)
        if self.graphIndex > self.maxIndex:
            self.graphIndex = self.maxIndex
        self.draw_graph(self.graphIndex)
    def move_down(self):
        # Need to call the correct draw, whether we're on graph one or two
        self.graphIndex = (self.graphIndex - 1)
        if self.graphIndex < 0:
            self.graphIndex = 0
        self.draw_graph(self.graphIndex)


root = Tk()
matplotlibSwitchGraphs(root)
root.mainloop()

