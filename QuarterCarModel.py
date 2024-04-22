#region imports
import sys
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from Car_GUI import Ui_Form

#these imports are necessary for drawing a matplot lib graph on my GUI
#no simple widget for this exists in QT Designer, so I have to add the widget in code.
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#endregion

#region class definitions
#region specialized graphic items
class MassBlock(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, width=30, height=10, parent=None, pen=None, brush=None, name='CarBody', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = brush
        self.width = width
        self.height = height
        self.top = self.y - self.height/2
        self.left = self.x - self.width/2
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()
        stTT = self.name +"\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)

    def boundingRect(self):
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)  # Red color pen
        if self.brush is not None:
            painter.setBrush(self.brush)
        self.top = -self.height/2
        self.left = -self.width/2
        self.rect=qtc.QRectF( self.left, self.top, self.width, self.height)
        painter.drawRect(self.rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        self.transformation.reset()
        # brPen=qtg.QPen()
        # brPen.setWidth(0)
        # painter.setPen(brPen)
        # painter.setBrush(qtc.Qt.NoBrush)
        # painter.drawRect(self.boundingRect())

class Wheel(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, radius=10, parent=None, pen=None, wheelBrush=None, massBrush=None, name='Wheel', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = wheelBrush
        self.radius = radius
        self.rect = qtc.QRectF(self.x - self.radius, self.y - self.radius, self.radius*2, self.radius*2)
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()
        stTT = self.name +"\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)
        self.massBlock = MassBlock(CenterX, CenterY, width=2*radius*0.85, height=radius/3, pen=pen, brush=massBrush, name="Wheel Mass", mass=mass)

    def boundingRect(self):
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect
    def addToScene(self, scene):
        scene.addItem(self)
        scene.addItem(self.massBlock)

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)  # Red color pen
        if self.brush is not None:
            painter.setBrush(self.brush)
        self.rect=qtc.QRectF(-self.radius, -self.radius, self.radius*2, self.radius*2)
        painter.drawEllipse(self.rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        self.transformation.reset()
        # brPen=qtg.QPen()
        # brPen.setWidth(0)
        # painter.setPen(brPen)
        # painter.setBrush(qtc.Qt.NoBrush)
        # painter.drawRect(self.boundingRect())

#endregion

#region MVC for quarter car model
class CarModel():
    """
    I re-wrote the quarter car model as an object oriented program
    and used the MVC pattern.  This is the quarter car model.  It just
    stores information about the car and results of the ode calculation.
    """
    def __init__(self):
        """
        self.results to hold results of odeint solution
        self.t time vector for odeint and for plotting
        self.tramp is time required to climb the ramp
        self.angrad is the ramp angle in radians
        self.ymag is the ramp height in m
        """
        self.results = []
        self.tmax = 3.0  # limit of timespan for simulation in seconds
        self.t = np.linspace(0, self.tmax, 200)
        self.tramp = 1.0  # time to traverse the ramp in seconds
        self.angrad = 0.1
        self.ymag = 6.0 / (12 * 3.3)  # ramp height in meters.  default is 0.1515 m
        self.yangdeg = 45.0  # ramp angle in degrees.  default is 45
        self.results = None

        #set default values for the properties of the quarter car model
        self.m1 = 450 # mass of car body in kg
        self.m2 = 15  # mass of wheel in kg
        self.c1 = 100  # starting guess for damping coefficient in N*s/m
        self.k1 = (self.m1 * 9.81) / (0.0762)  # spring constant of suspension in N/m, calculated for 3 inch compression
        self.k2 = (self.m2 * 9.81) / (0.01905)  # spring constant of tire in N/m, calculated for 0.75 inch compression
        self.v = 120  # velocity of car in kph from GUI input


        self.mink1 = (self.m1 * 9.81) / (0.0762)  # softer suspension, 3 inch compression
        self.maxk1 = (self.m1 * 9.81) / (0.0508)  # stiffer suspension 2 inch compression
        self.mink2 = (self.m2 * 9.81) / (0.0381)  # softer tires, 1.5 inch compression
        self.maxk2 = (self.m2 * 9.81) / (0.01905)  # stiffer tire, 0.75 inch compression
        self.accel =None
        self.accelMax = 2.0  # maximum acceleration allowed, 2.0g
        self.accelLim = 2.0  # acceleration limit, 2.0g
        self.SSE = 0.0

class CarView():
    def __init__(self, args):
        self.input_widgets, self.display_widgets = args
        # unpack widgets with same names as they have on the GUI
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
         self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        # creating a canvas to draw a figure for the car model
        self.figure = Figure(tight_layout=True, frameon=True, facecolor='none')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout_horizontal_main.addWidget(self.canvas)

        # axes for the plotting using view
        self.ax = self.figure.add_subplot()
        if self.ax is not None:
            self.ax1 = self.ax.twinx()

        self.buildScene()

    def updateView(self, model=None):
        self.le_m1.setText("{:0.2f}".format(model.m1))
        self.le_k1.setText("{:0.2f}".format(model.k1))
        self.le_c1.setText("{:0.2f}".format(model.c1))
        self.le_m2.setText("{:0.2f}".format(model.m2))
        self.le_k2.setText("{:0.2f}".format(model.k2))
        self.le_ang.setText("{:0.2f}".format(model.yangdeg))
        self.le_tmax.setText("{:0.2f}".format(model.tmax))
        stTmp="k1_min = {:0.2f}, k1_max = {:0.2f}\nk2_min = {:0.2f}, k2_max = {:0.2f}\n".format(model.mink1, model.maxk1, model.mink2, model.maxk2)
        stTmp+="SSE = {:0.2f}".format(model.SSE)
        self.lbl_MaxMinInfo.setText(stTmp)
        self.doPlot(model)

    def buildScene(self):
        #create a scene object
        self.scene = qtw.QGraphicsScene()
        self.scene.setObjectName("MyScene")
        self.scene.setSceneRect(-200, -200, 400, 400)  # xLeft, yTop, Width, Height

        #set the scene for the graphics view object
        self.gv_Schematic.setScene(self.scene)
        #make some pens and brushes for my drawing
        self.setupPensAndBrushes()
        self.Wheel = Wheel(0,50,50, pen=self.penWheel, wheelBrush=self.brushWheel, massBrush=self.brushMass, name = "Wheel")
        self.CarBody = MassBlock(0, -70, 100, 30, pen=self.penWheel, brush=self.brushMass, name="Car Body", mass=150)
        self.Wheel.addToScene(self.scene)
        self.scene.addItem(self.CarBody)
        self.gv_Schematic.setScene(self.scene)

    def setupPensAndBrushes(self):
        self.penWheel = qtg.QPen(qtg.QColor("orange"))
        self.penWheel.setWidth(1)
        self.brushWheel = qtg.QBrush(qtg.QColor.fromHsv(35,255,255, 64))
        self.brushMass = qtg.QBrush(qtg.QColor(200,200,200, 128))

    def doPlot(self, model=None):
        if model.results is None:
            return
        ax = self.ax
        ax1=self.ax1
        # plot result of odeint solver
        QTPlotting = True  # assumes we are plotting onto a QT GUI form
        if ax == None:
            ax = plt.subplot()
            ax1=ax.twinx()
            QTPlotting = False  # actually, we are just using CLI and showing the plot
        ax.clear()
        ax1.clear()
        t=model.t
        ycar = model.results[:,0]
        ywheel=model.results[:,2]
        accel=model.accel

        if self.chk_LogX.isChecked():
            ax.set_xlim(0.001,model.tmax)
            ax.set_xscale('log')
        else:
            ax.set_xlim(0.0, model.tmax)
            ax.set_xscale('linear')

        if self.chk_LogY.isChecked():
            ax.set_ylim(0.0001,max(ycar.max(), ywheel.max()*1.05))
            ax.set_yscale('log')
        else:
            ax.set_ylim(0.0, max(ycar.max(), ywheel.max()*1.05))
            ax.set_yscale('linear')

        ax.plot(t, ycar, 'b-', label='Body Position')
        ax.plot(t, ywheel, 'r-', label='Wheel Position')
        if self.chk_ShowAccel.isChecked():
            ax1.plot(t, accel, 'g-', label='Body Accel')
            ax1.axhline(y=accel.max(), color='orange')  # horizontal line at accel.max()
            ax1.set_yscale('log' if self.chk_LogAccel.isChecked() else 'linear')

        # add axis labels
        ax.set_ylabel("Vertical Position (m)", fontsize='large' if QTPlotting else 'medium')
        ax.set_xlabel("time (s)", fontsize='large' if QTPlotting else 'medium')
        ax1.set_ylabel("Y'' (g)", fontsize = 'large' if QTPlotting else 'medium')
        ax.legend()

        ax.axvline(x=model.tramp)  # vertical line at tramp
        ax.axhline(y=model.ymag)  # horizontal line at ymag
        # modify the tick marks
        ax.tick_params(axis='both', which='both', direction='in', top=True,
                       labelsize='large' if QTPlotting else 'medium')  # format tick marks
        ax1.tick_params(axis='both', which='both', direction='in', right=True,
                       labelsize='large' if QTPlotting else 'medium')  # format tick marks
        # show the plot
        if QTPlotting == False:
            plt.show()
        else:
            self.canvas.draw()

class CarController():
    def __init__(self, args):
        """
        This is the controller I am using for the quarter car model.
        """
        self.input_widgets, self.display_widgets = args
        self.model = CarModel()
        self.view = CarView(args)

        #self.chk_IncludeAccel=qtw.QCheckBox()

    def ode_system(self, X, t):
        # define the forcing function equation for the linear ramp
        # It takes self.tramp time to climb the ramp, so y position is
        # a linear function of time.
        if t < self.model.tramp:
            y = self.model.ymag * (t / self.model.tramp)
        else:
            y = self.model.ymag

        x1 = X[0]  # car position in vertical direction
        x1dot = X[1]  # car velocity  in vertical direction
        x2 = X[2]  # wheel position in vertical direction
        x2dot = X[3]  # wheel velocity in vertical direction

        # write the non-trivial equations in vertical direction
        x1ddot = (self.k2 * (y - x2) - self.k1 * (x1 - x2) - self.c1 * (x1dot - x2dot)) / self.m1
        x2ddot = (self.k1 * (x1 - x2) + self.c1 * (x1dot - x2dot) - self.k2 * (y - x2)) / self.m2

        # return the derivatives of the input state vector
        return [x1dot, x1ddot, x2dot, x2ddot]

    def calculate(self, doCalc=True):
        """
        I will first set the basic properties of the car model and then calculate the result
        in another function doCalc.
        """
            # Convert and validate input values
        try:
            # Fetch and validate GUI inputs
            self.model.m1 = float(self.input_widgets['le_m1'].text())
            self.model.m2 = float(self.input_widgets['le_m2'].text())
            self.model.v = float(self.input_widgets['le_v'].text())
            self.model.c1 = float(self.input_widgets['le_c1'].text())
            self.model.k1 = float(self.input_widgets['le_k1'].text())
            self.model.k2 = float(self.input_widgets['le_k2'].text())
            self.model.yangdeg = float(self.input_widgets['le_ang'].text())
            self.model.tmax = float(self.input_widgets['le_tmax'].text())

            # Validate ranges
            if not (0 < self.model.m1 < 10000):
                raise ValueError("Car body mass is out of acceptable range.")
            if not (0 < self.model.v < 300):
                raise ValueError("Velocity is out of acceptable range.")
            if not (0 < self.model.k1 < 50000):
                raise ValueError("Suspension spring constant is out of acceptable range.")
            if not (0 < self.model.k2 < 200000):
                raise ValueError("Tire spring constant is out of acceptable range.")
            if not (0 <= self.model.yangdeg <= 90):
                raise ValueError("Ramp angle is out of acceptable range.")
            if not (0 < self.model.tmax < 10):
                raise ValueError("Maximum plot time is out of acceptable range.")

            # Proceed with calculations if inputs are valid
            if doCalc:
                self.doCalc()
            self.view.updateView(self.model)
        except ValueError as e:
            QtWidgets.QMessageBox.critical(None, "Input Error", str(e))
            return


    def setWidgets(self, w):
        self.view.setWidgets(w)
        self.chk_IncludeAccel=self.view.chk_IncludeAccel

    def doCalc(self, doPlot=True, doAccel=True):
        """
        Ensure that all parameters are set and calculations are based on current inputs.
        """
        v = 1000 * self.model.v / 3600  # Convert speed to m/s from kph
        self.model.angrad = self.model.yangdeg * math.pi / 180  # Convert angle to radians
        self.model.tramp = self.model.ymag / (math.sin(self.model.angrad) * v)  # Time to traverse ramp

        self.model.t = np.linspace(0, self.model.tmax, 2000)
        ic = [0, 0, 0, 0]  # Initial conditions
        self.model.results = odeint(self.ode_system, ic, self.model.t)
        if doAccel:
            self.calcAccel()

        if doPlot:
            self.view.doPlot(self.model)
        pass
    def calcAccel(self):
        """
        Calculate the acceleration in the vertical direction using the forward difference formula.
        """
        N=len(self.model.t)
        self.model.accel=np.zeros(shape=N)
        vel=self.model.results[:,1]
        for i in range(N):
            if i==N-1:
                h = self.model.t[i] - self.model.t[i-1]
                self.model.accel[i]=(vel[i]-vel[i-1])/(9.81*h)  # backward difference of velocity
            else:
                h = self.model.t[i + 1] - self.model.t[i]
                self.model.accel[i] = (vel[i + 1] - vel[i]) / (9.81 * h)  # forward difference of velocity
            #else:
                #self.model.accel[i]=(vel[i+1]-vel[i-1])/(9.81*2.0*h)  # central difference of velocity
        self.model.accelMax=self.model.accel.max()
        return True

    def OptimizeSuspension(self):
        """
        Step 1:  set parameters based on GUI inputs by calling self.set(doCalc=False)
        Step 2:  make an initial guess for k1, c1, k2
        Step 3:  optimize the suspension
        :return:
        """
        try:
            self.calculate(doCalc=False)  # Setup model with GUI values
            initial_guesses = [self.model.k1, self.model.c1, self.model.k2]  # Use current model values
            result = minimize(self.SSE, initial_guesses, method='Nelder-Mead')
            if result.success:
                self.model.k1, self.model.c1, self.model.k2 = result.x
                self.doCalc()  # Recalculate with optimized values
                self.view.updateView(self.model)
            else:
                raise Exception("Optimization failed: " + result.message)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Optimization Error", str(e))

    def SSE(self, vals, optimizing=True):
        """
        Calculates the sum of square errors between the contour of the road and the car body.
        :param vals:
        :param optimizing:
        :return:
        """
        k1, c1, k2=vals  #unpack the new values for k1, c1, k2
        self.model.k1=k1
        self.model.c1=c1
        self.model.k2=k2
        self.doCalc(doPlot=False)  #solve the odesystem with the new values of k1, c1, k2
        SSE=0
        for i in range(len(self.model.results[:,0])):
            t=self.model.t[i]
            y=self.model.results[:,0][i]
            if t < self.model.tramp:
                ytarget = self.model.ymag * (t / self.model.tramp)
            else:
                ytarget = self.model.ymag
            SSE+=(y-ytarget)**2

        #some penalty functions if the constants are too small
        if optimizing:
            if k1<self.model.mink1 or k1>self.model.maxk1:
                SSE+=100
            if c1<10:
                SSE+=100
            if k2<self.model.mink2 or k2>self.model.maxk2:
                SSE+=100

            # I'm overlaying a gradient in the acceleration limit that scales with distance from a target squared.
            if self.model.accelMax>self.model.accelLim and self.chk_IncludeAccel.isChecked():
                # need to soften suspension
                SSE+=(self.model.accelMax-self.model.accelLim)**2
        self.model.SSE=SSE
        return SSE

    def doPlot(self):
        """
        Updated method to handle plotting based on the latest results.
        """
        if self.view.chk_LogX.isChecked() or self.view.chk_LogY.isChecked() or self.view.chk_LogAccel.isChecked():
            self.view.doPlot(self.model)
#endregion
#endregion

def main():
    app = qtw.QApplication(sys.argv)
    MainWindow = qtw.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    # Prepare args for CarController
    input_widgets = {
        'le_m1': ui.le_m1,
        'le_v': ui.le_v,
        'le_k1': ui.le_k1,
        'le_c1': ui.le_c1,
        'le_m2': ui.le_m2,
        'le_k2': ui.le_k2,
        'le_ang': ui.le_ang,
        'le_tmax': ui.le_tmax,
        'chk_IncludeAccel': ui.chk_IncludeAccel
    }
    display_widgets = {
        'gv_Schematic': ui.gv_Schematic,
        'chk_LogX': ui.chk_LogX,
        'chk_LogY': ui.chk_LogY,
        'chk_LogAccel': ui.chk_LogAccel,
        'chk_ShowAccel': ui.chk_ShowAccel,
        'lbl_MaxMinInfo': ui.lbl_MaxMinInfo,
        'layout_horizontal_main': ui.layout_horizontal_main
    }

    args = (input_widgets, display_widgets)
    QCM = CarController(args)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
