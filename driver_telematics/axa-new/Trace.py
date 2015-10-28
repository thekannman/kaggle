# $Id: Trace.py,v 1.2 2015/03/11 05:31:37 zak Exp zak $
import os
from math import hypot
import numpy as np

import warnings
warnings.filterwarnings('error')

np.seterr(all='print')

vcorr = 25
acorr = 25

def reject_outliers(data, m=10):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def list_outliers(data, m=10):
    return (abs(data - np.mean(data)) > m * np.std(data))

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
    return np.array(xnew), np.array(ynew)


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
    adotv = np.zeros(len(x)-2,dtype=float)
    adotvhat = np.zeros(len(x)-2,dtype=float)
    #vacf = np.zeros(vcorr,dtype=float)
    count = np.zeros(vcorr,dtype=float)
    #vacf_int = 1.0
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
    #for i in range(len(dx)):
    #    stop = min(i+vcorr*5,len(dx))
    #    for j in range(i,stop,5):
    #        vacf[(j-i)/5] += dx[i]*dx[j] + dy[i]*dy[j]
    #        count[(j-i)/5] += 1
    #if vacf[0] == 0.0:
    #    for i in range(len(dx)-vcorr):
    #        print dx[i]*dx[i], dy[i]*dy[i]
    #    exit()
    #vacf[0] = vacf[0] / count[0]
    #for i in range(1,vcorr):
    #    vacf[i] = vacf[i] / vacf[0] / count[i]
    #    vacf_int += vacf[i]
    #vacf[0] = 1.0
    for i in xrange(1, len(dx)):
        dvx = dx[i] - dx[i-1]
        dvy = dy[i] - dy[i-1]
        adotv[i-1] = dvx*(dx[i] + dx[i-1])/2.0 + dvy*(dy[i] + dy[i-1])/2.0
        speed = distance(dx[i-1], dy[i-1], dx[i], dy[i])
        if not (v[i-1] == 0.0 and v[i] == 0.0):
          adotvhat[i-1] = adotv[i-1]/((v[i-1]+v[i])/2.0) # Not yet incorporated.
        else:  
          adotvhat[i-1] = 0.0
        a[i-1] = speed
    return a, v, distancesum, stopfreq, adotv, adotvhat#,vacf,vacf_int


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
        #a, v, self.distancecovered,stopfreq,self.vacf,self.vacf_int = velocities_and_distance_covered(self.__xn, self.__yn)
	a, v, self.distancecovered,stopfreq, adotv, adotvhat = velocities_and_distance_covered(self.__xn, self.__yn)
        #a = reject_outliers(a)
        #v = reject_outliers(v)
        #adotv = reject_outliers(adotv)
        #adotvhat = reject_outliers(adotvhat)
        self.adotv = adotv
        #v_w = np.fft.rfft(v).imag
	#a_w = np.fft.rfft(a).imag
        #self.vibspec = np.fft.rfft(self.vacf).imag
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
        self.maxadotv = np.max(adotv)
        self.adotv10,self.adotv20,self.adotv30,self.adotv40,self.adotv60,self.adotv70,self.adotv80,self.adotv90 = np.percentile(adotv,[10,20,30,40,60,70,80,90])
        self.avgadotv = np.average(adotv)
        self.medadotv = np.median(adotv)
        self.stdadotv = np.std(adotv)
        self.minadotv = np.min(adotv)
        self.maxadotvhat = np.max(adotvhat)
        self.adotvhat10,self.adotvhat20,self.adotvhat30,self.adotvhat40,self.adotvhat60,self.adotvhat70,self.adotvhat80,self.adotvhat90 = np.percentile(adotvhat,[10,20,30,40,60,70,80,90])
        self.avgadotvhat = np.average(adotvhat)
        self.medadotvhat = np.median(adotvhat)
        self.stdadotvhat = np.std(adotvhat)
        self.minadotvhat = np.min(adotvhat)
	self.stopfreq = stopfreq
	#self.max_vibspec = np.max(self.vibspec)
	#self.max_vibspec_freq = np.argmax(self.vibspec)
	#self.min_vibspec = np.min(self.vibspec)
	#self.min_vibspec_freq = np.argmin(self.vibspec)
	#self.max_a_w = np.max(a_w)
	#self.max_a_w_freq = np.argmax(a_w)
	#self.min_a_w = np.min(a_w)
	#self.min_a_w_freq = np.argmin(a_w) 
        self.triplength = distance(x[0], y[0], x[-1], y[-1])
        self.triptime = len(x)
        #distances = [distance(self.__xn[0], self.__yn[0], self.__xn[i], self.__yn[i]) for i in range(len(self.__xn))]
        #distance_outliers = list_outliers(distances)
        #self.__xn = self.__xn[np.logical_not(distance_outliers)]
        #self.__yn = self.__yn[np.logical_not(distance_outliers)]
        #self.scaledX = np.zeros(100,dtype=float) 
        #self.scaledY = np.zeros(100,dtype=float)
        #if self.triplength == 0.0:
        #    for i in range(100):
        #        self.scaledX[i] = 0.0
        #        self.scaledY[i] = 0.0
        #else:
        #    for i in range(100):
        #        self.scaledX[i] = self.__xn[int(i/100.0*(len(self.__xn)-1))]/self.triplength
        #        self.scaledY[i] = self.__yn[int(i/100.0*(len(self.__yn)-1))]/self.triplength
        #self.Xaverage = np.average(self.__xn)
        #self.Yaverage = np.average(self.__yn)
        #self.scaledXaverage = np.average(self.scaledX)
        #self.scaledYaverage = np.average(self.scaledY)
        #self.Xstd = np.std(self.__xn)
        #self.Ystd = np.std(self.__yn)
        #self.scaledXstd = np.std(self.scaledX)
        #self.scaledYstd = np.std(self.scaledY)
        #self.Xpercentiles = np.percentile(self.__xn,[0,10,20,30,40,60,70,80,90,100])
        #self.Ypercentiles = np.percentile(self.__yn,[0,10,20,30,40,60,70,80,90,100])
        #self.scaledXpercentiles = np.percentile(self.scaledX,[0,10,20,30,40,60,70,80,90,100])
        #self.scaledYpercentiles = np.percentile(self.scaledY,[0,10,20,30,40,60,70,80,90,100])
        #self.distAverage = np.average(distances)
        #self.distStd = np.std(distances)
        #self.disPpercentiles = np.percentile(distances,[0,10,20,30,40,60,70,80,90,100])
        #self.scaledDistances = [distance(self.scaledX[0], self.scaledY[0], self.scaledX[i], self.scaledY[i]) for i in range(100)]
        #self.scaledDistAverage = np.average(self.scaledDistances)
        #self.scaledDistStd = np.std(self.scaledDistances)
        #self.scaledDistPercentiles = np.percentile(self.scaledDistances,[0,10,20,30,40,60,70,80,90,100])

    @property
    def features(self):
        """Returns a list that comprises all computed features of this trace."""
        features = []
        features.append(self.triplength)
        features.append(self.triptime)
        features.append(self.distancecovered)
        features.append(self.triplength/self.triptime)
        features.append(self.triplength/self.distancecovered)
        features.append(self.triptime/self.distancecovered)
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

        features.append(self.avgspeed/self.maxspeed)
        features.append(self.speed10/self.maxspeed)
        features.append(self.speed20/self.maxspeed)
        features.append(self.speed30/self.maxspeed)
        features.append(self.speed40/self.maxspeed)
        features.append(self.speed60/self.maxspeed)
        features.append(self.speed70/self.maxspeed)
        features.append(self.speed80/self.maxspeed)
        features.append(self.speed90/self.maxspeed)
        features.append(self.medspeed/self.maxspeed)
        features.append(self.stdspeed/self.maxspeed)

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

        features.append(self.avgaccel/self.maxaccel)
        features.append(self.accel10/self.maxaccel)
        features.append(self.accel20/self.maxaccel)
        features.append(self.accel30/self.maxaccel)
        features.append(self.accel40/self.maxaccel)
        features.append(self.accel60/self.maxaccel)
        features.append(self.accel70/self.maxaccel)
        features.append(self.accel80/self.maxaccel)
        features.append(self.accel90/self.maxaccel)
        features.append(self.medaccel/self.maxaccel)
        features.append(self.stdaccel/self.maxaccel)

        features.append(self.maxadotv)
        features.append(self.avgadotv)
        features.append(self.adotv10)
        features.append(self.adotv20)
        features.append(self.adotv30)
        features.append(self.adotv40)
        features.append(self.adotv60)
        features.append(self.adotv70)
        features.append(self.adotv80)
        features.append(self.adotv90)
        features.append(self.medadotv)
        features.append(self.stdadotv)

        features.append(self.avgadotv/self.maxadotv)
        features.append(self.adotv10/self.maxadotv)
        features.append(self.adotv20/self.maxadotv)
        features.append(self.adotv30/self.maxadotv)
        features.append(self.adotv40/self.maxadotv)
        features.append(self.adotv60/self.maxadotv)
        features.append(self.adotv70/self.maxadotv)
        features.append(self.adotv80/self.maxadotv)
        features.append(self.adotv90/self.maxadotv)
        features.append(self.medadotv/self.maxadotv)
        features.append(self.stdadotv/self.maxadotv)

        features.append(self.maxadotvhat)
        features.append(self.avgadotvhat)
        features.append(self.adotvhat10)
        features.append(self.adotvhat20)
        features.append(self.adotvhat30)
        features.append(self.adotvhat40)
        features.append(self.adotvhat60)
        features.append(self.adotvhat70)
        features.append(self.adotvhat80)
        features.append(self.adotvhat90)
        features.append(self.medadotvhat)
        features.append(self.stdadotvhat)

        features.append(self.avgadotvhat/self.maxadotvhat)
        features.append(self.adotvhat10/self.maxadotvhat)
        features.append(self.adotvhat20/self.maxadotvhat)
        features.append(self.adotvhat30/self.maxadotvhat)
        features.append(self.adotvhat40/self.maxadotvhat)
        features.append(self.adotvhat60/self.maxadotvhat)
        features.append(self.adotvhat70/self.maxadotvhat)
        features.append(self.adotvhat80/self.maxadotvhat)
        features.append(self.adotvhat90/self.maxadotvhat)
        features.append(self.medadotvhat/self.maxadotvhat)
        features.append(self.stdadotvhat/self.maxadotvhat)
        
        #features.extend(self.scaledX)
        #features.extend(self.scaledY)
        #features.append(self.Xaverage)
        #features.append(self.Yaverage)
        #features.append(self.scaledXaverage)
        #features.append(self.scaledYaverage)
        #features.append(self.Xstd)
        #features.append(self.Ystd)
        #features.append(self.scaledXstd)
        #features.append(self.scaledYstd)
        #features.extend(self.Xpercentiles)
        #features.extend(self.Ypercentiles)
        #features.extend(self.scaledXpercentiles)
        #features.extend(self.scaledYpercentiles)
        #features.append(self.distAverage)
        #features.append(self.distStd)
        #features.extend(self.disPpercentiles)
        #features.extend(self.scaledDistances)
        #features.append(self.scaledDistAverage)
        #features.append(self.scaledDistStd)
        #features.extend(self.scaledDistPercentiles)

	#features.append(self.max_vibspec)
        #features.append(self.max_vibspec_freq)
        #features.append(self.min_vibspec)
        #features.append(self.min_vibspec_freq)
        #features.append(self.max_a_w)
        #features.append(self.max_a_w_freq)
        #features.append(self.min_a_w)
        #features.append(self.min_a_w_freq)
        #features.append(self.vacf_int)
        #features.extend(self.vibspec)
        #features.extend(self.vacf)
        if np.sum(np.isnan(features)):
            print self.adotv
            exit()
        return features

    def __str__(self):
        return "Trace {0} has this many positions: \n {1}".format(self.__id, self.triptime)

    @property
    def identifier(self):
        """Driver identifier is its filename"""
        return self.__id
