import os
from math import hypot
import numpy as np

vcorr = 125
acorr = 125

def smooth(x, y, steps):
    """
    Returns moving average using steps samples to generate the new trace

    Input: x-coordinates and y-coordinates as lists as well as an integer to indicate the size of the window (in steps)
    Output: list for smoothed x-coordinates and y-coordinates
    """
    xnew = []
    ynew = []
    for i in xrange(steps, len(x)):
        xnew.append(sum(x[i-steps:i]) / float(steps))
        ynew.append(sum(y[i-steps:i]) / float(steps))
    return xnew, ynew


def distance(x0, y0, x1, y1):
    """Computes 2D euclidean distance"""
    return hypot((x1 - x0), (y1 - y0))


def velocities_and_distance_covered(x, y):
    """
    Returns velocities just using difference in distance between coordinates as well as accumulated distances

    Input: x-coordinates and y-coordinates as lists
    Output: list of velocities
    """
    v = np.zeros(len(x)-1,dtype=float)
    dx = np.zeros(len(x)-1,dtype=float)
    dy = np.zeros(len(x)-1,dtype=float)
    a = np.zeros(len(x)-2,dtype=float)
    vacf = np.zeros(vcorr,dtype=float)
    count = np.zeros(vcorr,dtype=float)
    vacf_int = 1.0
    distancesum = 0.0
    stopcount = 0
    for i in xrange(1, len(x)):
	dx[i-1] = x[i] - x[i-1]
	dy[i-1] = y[i] - y[i-1]
        dist = distance(x[i-1], y[i-1], x[i], y[i])
        v[i-1] = dist
	if v[i-1] == 0:
            stopcount += 1
        distancesum += dist
    stopfreq = float(stopcount)/float(len(x)-1)
    for i in range(len(dx)):
        stop = min(i+vcorr,len(dx))
        for j in range(i,stop):
            vacf[j-i] += dx[i]*dx[j] + dy[i]*dy[j]
            count[j-i] += 1
    if vacf[0] == 0.0:
        for i in range(len(dx)-vcorr):
            print dx[i]*dx[i], dy[i]*dy[i]
        exit()
    vacf[0] = vacf[0] / count[0]
    for i in range(1,vcorr):
        vacf[i] = vacf[i] / vacf[0] / count[i]
        vacf_int += vacf[i]
    vacf[0] = 1.0
    for i in xrange(1, len(dx)):
        speed = distance(dx[i-1], dy[i-1], dx[i], dy[i])
        a[i-1] = speed
    return a, v, distancesum, stopfreq,vacf,vacf_int


class Trace(object):
    """"
    Trace class reads a trace file and computes features of that trace.
    """

    def __init__(self, filename, filtering=10):
        """Input: path and name of the file of a trace; how many filtering steps should be used for sliding window filtering"""
        self.__id = int(os.path.basename(filename).split(".")[0])
        x = []
        y = []
        with open(filename, "r") as trainfile:
            trainfile.readline()  # skip header
            for line in trainfile:
                items = line.split(",", 2)
                x.append(float(items[0]))
                y.append(float(items[1]))
        self.__xn, self.__yn = smooth(x, y, filtering)
        a, v, self.distancecovered,stopfreq,self.vacf,self.vacf_int = velocities_and_distance_covered(self.__xn, self.__yn)

        #v_w = np.fft.rfft(v).imag
	#a_w = np.fft.rfft(a).imag
        #self.vibspec = np.fft.rfft(vacf).imag
	self.maxspeed = np.max(v)
	self.speed10,self.speed20,self.speed30,self.speed40,self.speed60,self.speed70,self.speed80,self.speed90 = np.percentile(v,[10,20,30,40,60,70,80,90])
	self.medspeed = np.median(v)
	self.stdspeed = np.std(v)
	self.avgspeed = np.average(v)
	self.maxaccel = np.max(a)
	self.accel10,self.accel20,self.accel30,self.accel40,self.accel60,self.accel70,self.accel80,self.accel90 = np.percentile(a,[10,20,30,40,60,70,80,90])
	self.avgaccel = np.average(a)
	self.medaccel = np.median(a)
	self.stdaccel = np.std(a)
	self.maxbreak = np.min(a)
	self.stopfreq = stopfreq
	#self.max_v_w = np.max(v_w)
	#self.max_v_w_freq = np.argmax(v_w)
	#self.min_v_w = np.min(v_w)
	#self.min_v_w_freq = np.argmin(v_w)
	#self.max_a_w = np.max(a_w)
	#self.max_a_w_freq = np.argmax(a_w)
	#self.min_a_w = np.min(a_w)
	#self.min_a_w_freq = np.argmin(a_w) 
        self.triplength = distance(x[0], y[0], x[-1], y[-1])
        self.triptime = len(x)

    @property
    def features(self):
        """Returns a list that comprises all computed features of this trace."""
        features = []
        features.append(self.triplength)
        features.append(self.triptime)
        features.append(self.distancecovered)
        features.append(self.maxspeed)
	features.append(self.avgspeed)
	features.append(self.maxaccel)
	features.append(self.avgaccel)
	features.append(self.maxbreak)
	features.append(self.stopfreq)
	features.append(self.speed10)
        features.append(self.speed20)
        features.append(self.speed30)
        features.append(self.speed40)
        features.append(self.speed60)
        features.append(self.speed70)
        features.append(self.speed80)
        features.append(self.speed90)
	features.append(self.medspeed)
	features.append(self.stdspeed)
	features.append(self.accel10)
        features.append(self.accel20)
        features.append(self.accel30)
        features.append(self.accel40)
        features.append(self.accel60)
        features.append(self.accel70)
        features.append(self.accel80)
        features.append(self.accel90)
	features.append(self.medaccel)
	features.append(self.stdaccel)
	#features.append(self.max_v_w)
        #features.append(self.max_v_w_freq)
        #features.append(self.min_v_w)
        #features.append(self.min_v_w_freq)
        #features.append(self.max_a_w)
        #features.append(self.max_a_w_freq)
        #features.append(self.min_a_w)
        #features.append(self.min_a_w_freq)
        features.append(self.vacf_int)
        #features.extend(self.vibspec)
        features.extend(self.vacf)
        return features

    def __str__(self):
        return "Trace {0} has this many positions: \n {1}".format(self.__id, self.triptime)

    @property
    def identifier(self):
        """Driver identifier is its filename"""
        return self.__id
