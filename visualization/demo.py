import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyads

from PyQt5.QtWidgets import (
   QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

mpl.rcParams['toolbar'] = 'None'

# Configuration constants
IP_ADR = '127.0.0.1.1.1'
SIMULATE_ADS_CONN = False

class AnimationCanvas(FigureCanvas):
    def __init__(self, parent=None, hide_title_bar=False, figure_bg_color='black', axes_bg_color='white'):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 6))
        super(AnimationCanvas, self).__init__(self.fig)
        self.setParent(parent)

        self.ax.set_facecolor(axes_bg_color)
        self.fig.patch.set_facecolor(figure_bg_color)

        if hide_title_bar:
            parent.setWindowFlags(Qt.FramelessWindowHint)

        # Initialize PLC connection if not simulating
        if not SIMULATE_ADS_CONN:
            # Initialize PLC connection
            self.plc = pyads.Connection(IP_ADR, pyads.PORT_TC3PLC1)
            self.plc.open()

            # Configuration and initial setup
            self.num_tiles = np.array(self.plc.read_by_name("GVL.aNumTilesXY"))
            self.num_movers = len(self.plc.read_by_name("GVL.aMoverGoalsXSim"))
        else:
            self.num_tiles = np.array([4, 3])
            self.num_movers = 1

        self.EXIT_PRESS = False
        self.mover_color = ['blue','#fcc203', 'red', 'green', ]
        self.tile_size = np.array([0.12, 0.12]) # half-size!

        self.setup_plot()

        self.artists = {'goals': []}

        # Initialize the animation
        self.ani = FuncAnimation(
            self.fig,
            self.animate,
            frames=self.frame_generator(),
            init_func=self.init_animation,
            save_count=1000,
            interval=1,
            blit=False
        )

    def setup_plot(self):
        num_tiles_x = self.num_tiles[0]
        num_tiles_y = self.num_tiles[1]
        max_x = num_tiles_x * (self.tile_size[0] * 2)
        max_y = num_tiles_y * (self.tile_size[1] * 2)
        x_ticks = np.linspace(0, num_tiles_x * (self.tile_size[0] * 2), num=num_tiles_x + 1, endpoint=True)
        y_ticks = np.linspace(0, num_tiles_y * (self.tile_size[1] * 2), num=num_tiles_y + 1, endpoint=True)

        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        self.ax.set_xlim([np.min(x_ticks), np.max(x_ticks)])  # type: ignore
        self.ax.set_ylim([np.min(y_ticks), np.max(y_ticks)])  # type: ignore
        # lines
        for x in x_ticks:
            self.ax.plot([x,x], [0,max_y], '-k')
        for y in y_ticks:
            self.ax.plot([0,max_x], [y,y], '-k')

        self.ax.xaxis.set_inverted(True)
        self.ax.yaxis.set_inverted(True)
        self.ax.set_aspect('equal')
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax.set_facecolor('white')
        # minimize margins
        self.fig.tight_layout()

    def read_state(self, frame):
        # This function should return the state array similar to the one you're currently fetching in the loop
        if SIMULATE_ADS_CONN:
            state = [0] * 6 * self.num_movers
            for i in range(self.num_movers):
                state[0 + 6 * i] = 0.1 + 0.1 * np.cos(frame / 5)
                state[1 + 6 * i] = 0.1 + 0.1 * np.sin(frame / 5)
                state[2 + 6 * i] = -np.sin(frame / 5)
                state[3 + 6 * i] = np.cos(frame / 5)
                state[4 + 6 * i] = 0.05
                state[5 + 6 * i] = 0.05
            state = np.asarray(state)[:, np.newaxis]
        else:
            state = np.array(self.plc.read_by_name("GVL.aSimState"))
        return state

    def animate(self, frame):
        (index, artists) = frame
            
        state = self.read_state(index)
        self.draw_artists(state, artists, initial_draw=False)
        return self.ax,

    def init_animation(self):
        state = self.read_state(0)
        self.draw_artists(state, self.artists, initial_draw=True)
        return self.ax,

    def draw_artists(self, state, artists, initial_draw=False):
        # Draw or update the artists based on the current state
        if state.any():
            for idx_mover in range(0, self.num_movers):
                start_idx = (self.num_movers*2) + idx_mover * 2
                end_idx = start_idx + 2
                goal = state[start_idx:end_idx]
                if initial_draw:
                    # Draw goal marker
                    goal_marker, = self.ax.plot(
                        goal[0],
                        goal[1],
                        marker='x',
                        ms=50,
                        mew=15,
                        lw=4,
                        color=self.mover_color[idx_mover % len(self.mover_color)],
                        zorder=6
                    )
                    artists['goals'].append(goal_marker)
                elif idx_mover < len(self.artists['goals']):
                    # Update goal marker position
                    artists['goals'][idx_mover].set_xdata([goal[0]])
                    artists['goals'][idx_mover].set_ydata([goal[1]])

        self.ax.set_aspect('equal')
        self.draw()

    def frame_generator(self):
        # Create animation
        frame_cnt = 0
        while True:
            frame_cnt += 1
            yield frame_cnt, self.artists

    def set_figure_background_color(self, color):
        self.fig.patch.set_facecolor(color)
        self.draw()

    def set_axes_background_color(self, color):
        self.ax.set_facecolor(color)
        self.draw()

    def close_connection(self):
        # Close ads connection
        if not SIMULATE_ADS_CONN and hasattr(self, 'plc'):
            self.plc.close()

class MainWindow(QMainWindow):
    def __init__(self, hide_title_bar=False, figure_bg_color='black', axes_bg_color='white'):
        super(MainWindow, self).__init__()

        self.setStyleSheet("background-color: black;")

        # Optionally hide the title bar
        if hide_title_bar:
            self.setWindowFlags(Qt.FramelessWindowHint)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set up the layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Create the animation canvas
        self.animation_canvas = AnimationCanvas(self, 
                                                hide_title_bar=hide_title_bar,
                                                figure_bg_color=figure_bg_color,
                                                axes_bg_color=axes_bg_color)
        self.layout.addWidget(self.animation_canvas)

    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key_F:
            self.showMaximized()
        if a0.key() == Qt.Key_Escape:
            self.close()

    def mousePressEvent(self, a0):
        if a0.button() == Qt.LeftButton:
            self.offset = a0.pos()
        else:
            super().mousePressEvent(a0)

    def mouseMoveEvent(self, a0):
        if self.offset is not None and a0.buttons() == Qt.LeftButton:
            self.move(self.pos() + a0.pos() - self.offset)
        else:
            super().mouseMoveEvent(a0)

    def mouseReleaseEvent(self, a0):
        self.offset = None
        super().mouseReleaseEvent(a0)

    def closeEvent(self, event):
        self.animation_canvas.close_connection()
        event.accept()

def main():
    app = QApplication(sys.argv)

    # Set to True to hide the title bar, False to show it
    hide_title_bar = True

    window = MainWindow(hide_title_bar=hide_title_bar, figure_bg_color='black', axes_bg_color='white')
    window.show()
    
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()