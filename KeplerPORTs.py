"""
KeplerPORTs.py - Illustrate making use of numerous Kepler Planet Occurrence
    Rate Data Products for Data Release 25 and SOC 9.3 Kepler Pipeline version.
    This code generates a detection contour according to the documentation
    Burke, C.J. & Catanzarite, J. 2017, "Planet Detection Metrics: 
       Per-Target Detection Contours for Data Release 25", KSCI-19111-001
    Additional recommended background reading
    -Earlier Data Release 24 version of detection contour described in 
    Burke et al. 2015, ApJ, 809, 8
    -Transit injection and recovery tests for the Kepler pipeline
    Christiansen et al. 2013, ApJS, 207, 35
    Christiansen et al. 2015, ApJ, 810, 95   (One Year Kepler data)
    Christiansen et al. 2016, ApJ, 828, 99   (Data Release 24)
    Christiansen, J. L. 2017, Planet Detection Metrics: Pixel-Level Transit
            Injection Tests of Pipeline Detection Efficiency
            for Data Release 25 (KSCI-19110-001)
    Burke & Catanzarite 2017, Planet Detection Metrics: Per-Target Flux-Level
            Transit Injection Tests of TPS for Data Release 25 (KSCI-19109-001)
    -Kepler Target Noise and Data Quality metrics
    Burke & Catanzarite 2016, Planet Detection Metrics: Window and 
            One-Sigma Depth Functions for Data Release 25 (KSCI-19101-002)
    
    Assumes python packages numpy, scipy, matplotlib, astropy, and h5py are available
      and files 
      detectEffData_alpha12_02272017.h5
      detectEffData_alpha12_SlopeLongShort_02272017.txt
      detectEffData_alpha_base_02272017.txt
      kplr003429335_dr25_onesigdepth.fits
      kplr003429335_dr25_window.fits
      are available in the same directory as KeplerPORTs.py
    Invocation: python KeplerPORTs.py
    Output: Displays a series of figures and generates hardcopy

Notices:
Copyright 2017 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
NASA acknowledges the SETI Institute's primary role in authoring and producing the KeplerPORTs (Kepler Planet Occurrence Rate Tools) under Cooperative Agreement Number NNX13AD01A.

Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
"""
import numpy as np
import scipy.interpolate as interp
import scipy.stats as stat
import os.path
import h5py
from astropy.io import fits
import matplotlib.pyplot as plt

def mstar_from_stellarprops(rstar, logg):
    """Gives stellar mass from the rstar and logg
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
       OUTPUT:
         mstar - stellar mass [Msun]
    """
    # Convert logg and rstar into stellar mass assuming logg_sun=4.437
    mstar = 10.0**logg * rstar**2. / 10.0**4.437
    return mstar

def transit_duration(rstar, logg, per, ecc):
    """Transit duration
       assuming uniform distribution of cos(inc) orbits,
       assuming rstar/a is small, and assuming rp/rstar is small.
       INPUT:
        rstar - Radius of star [Rsun]
        logg - log surface gravity [cgs]
        per - Period of orbit [day]
        ecc - Eccentricity; hardcoded to be < 0.99 
       OUTPUT:
        durat - Transit duration [hr]
       COMMENTS:  example:  x=transit_duration(1.0,4.437,365.25,0.0)
                            x=10.19559 [hr] duration for planet in 1 year orbit
                            around sun
    """
    # Replace large ecc values with 0.99
    ecc = np.where(ecc > 0.99, 0.99, ecc)
    # Convert logg and rstar into stellar mass
    mstar = mstar_from_stellarprops(rstar, logg)
    # Semi-major axis in AU
    semia = mstar**(1.0/3.0) * (per/365.25)**(2.0/3.0)
    # transit duration e=0 including pi/4 effect of cos(inc) dist
    r_sun = 6.9598e10 # cm
    au2cm = 1.49598e13 # 1 AU = 1.49598e13 cm
    durat = (per*24.0) / 4.0 * (rstar*r_sun) / (semia*au2cm)
    #transit duration e > 0
    durat = durat * np.sqrt(1.0-ecc**2);

    return durat

def transit_duration_zero(rstar, logg, per, ecc):
    """Transit duration at zero impact parameter
       assuming rstar/a is small, and assuming rp/rstar is small.
       INPUT:
        rstar - Radius of star [Rsun]
        logg - log surface gravity [cgs]
        per - Period of orbit [day]
        ecc - Eccentricity; hardcoded to be < 0.99 
       OUTPUT:
        durat - Transit duration [hr]
    """
    # Replace large ecc values with 0.99
    ecc = np.where(ecc > 0.99, 0.99, ecc)
    # Convert logg and rstar into stellar mass
    mstar = mstar_from_stellarprops(rstar, logg)
    # Semi-major axis in AU
    semia = mstar**(1.0/3.0) * (per/365.25)**(2.0/3.0)
    # transit duration e=0 including pi/4 effect of cos(inc) dist
    r_sun = 6.9598e10 # cm
    au2cm = 1.49598e13 # 1 AU = 1.49598e13 cm
    durat = (per*24.0) / np.pi * (rstar*r_sun) / (semia*au2cm)
    #transit duration e > 0
    durat = durat * np.sqrt(1.0-ecc**2);

    return durat


def calc_density(rstar, mstar):
    """From rstar & mstar calculate average density [cgs]
       INPUT:
         rstar - Radius of star [Rsun]
         mstar - Mass of star [Msun]
       OUTPUT:
         density - Average density [g/cm^3]
    """
    # Assuming density sun = 1.408
    density = 1.408 * mstar / rstar**3.0
    return density

def calc_logg(rstar, mstar):
    """From rstar & mstar calculate logg [cgs]
       INPUT:
         rstar - Radius of star [Rsun]
         mstar - Mass of star [Msun]
       OUTPUT:
         logg - log surface gravity [g/cm^3]
    """
    # Assuming logg sun = 4.437
    logg = 4.437 + np.log10(mstar) - (2.0*np.log10(rstar))
    return logg

def depth_to_rp(rstar, depth):
    """Gives Planet radius [Rear] for a given depth of transit and rstar
       INPUT:
         rstar - Radius of star [Rsun]
         depth - Depth of transit [ppm]
       OUTPUT:
         rp - Radius of planet ***[Rear]***
    """
    # Planet radius assuming Rp=1 Rear and Rstar=1 Rsun has depth 84 ppm    
    rp = np.sqrt(depth/84.0) * rstar
    return rp

def rp_to_depth(rstar, rp):
    """Gives Planet radius [Rear] for a given depth of transit and rstar
       INPUT:
         rstar - Radius of star [Rsun]
         rp - Radius of planet ***[Rear]*** 
       OUTPUT:
         depth - Depth of transit [ppm]
    """
    # Depth of transit assuming Rp=1 Rear and Rstar=1 Rsun has depth 84 ppm    
    depth = 84.0 * rp**2.0 / rstar**2
    return depth


def midpoint_transit_depth(rstar, rp, limbcoeffs):
    """Provides exact depth of transit [ppm] for a midpoint cross (e.g. b=0)
        transit event.  Based upon the mathematical description of
        Mandel & Agol (2002)
        INPUT:
        rstar [Rsun] - Stellar radius
        rp [***Rearth***] - Planet radius in Earth radii
        limbcoeffs - numpy array of limb darkening coeffs
                        size 4 = four parameter law
                        size 3 = quadratic law
                        size 1 = linear law
        OUTPUT:
        depth [ppm] - Transit depth at b=0
    """
    lcs = np.copy(limbcoeffs)
    if limbcoeffs.size == 2:
        lcs = np.array([0.0, limbcoeffs[0] + 2.0*limbcoeffs[1], 0.0, \
                        -limbcoeffs[1]])
    if limbcoeffs.size == 1:
        lcs = np.array([0.0, limbcoeffs[0], 0.0, 0.0])
    rearthDrsun = 6378137.0/696000000.0
    krp = rp / rstar * rearthDrsun
    c0  = 1.0 - np.sum(lcs)
    aux1 = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    aux2 = np.concatenate(([c0], lcs)) / aux1
    omega = np.sum(aux2)
    ksq = 1.0 - krp * krp
    tmp0 = aux2[0] * ksq
    tmp1 = aux2[1] * np.power(ksq, 5.0/4.0)
    tmp2 = aux2[2] * np.power(ksq, 3.0/2.0)
    tmp3 = aux2[3] * np.power(ksq, 7.0/4.0)
    tmp4 = aux2[4] * ksq * ksq
    return (1.0 - (tmp0 + tmp1 + tmp2 + tmp3 + tmp4)/omega) * 1.0e6


def earthflux_at_period(rstar, logg, teff, period):
    """Gives equivalent solar-earth bolometric flux for a given period
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         teff - Effective Temperature [K]
         period - Orbital Period [day]
       OUTPUT:
         flx - Flux relative to sun-earth 
    """
    mstar = mstar_from_stellarprops(rstar, logg)
    # Calculate semi-major axis [AU]
    semia = mstar**(1.0/3.0) * (period/365.25)**(2.0/3.0)
    # Star bolometric luminosity in Lsun assuming teff_sun=5778.0
    lumstar = rstar**2.0 * (teff/5778.0)**4.0
    # Calculate solar earth bolometric flux ratio
    flx = lumstar / semia**2.0
    return flx

def period_at_earthflux(rstar, logg, teff, seff):
    """Gives period for a given equivalent solar-earth bolometric flux
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         teff - Effective Temperature [K]
         seff - insolation flux relative to sun-earth flux
       OUTPUT:
         period - Orbital period [days]
    """
    mstar = mstar_from_stellarprops(rstar, logg)
    # Calculate semi-major axis [AU] assuming teff_sun=5778.0
    semia = rstar * (teff/5778.0)**2 / np.sqrt(seff)
    period = ( semia / (mstar**(1.0/3.0)) )**(3.0/2.0) * 365.25
    return period

def period_from_teq(rstar, logg, teff, teq, f, alb):
    """Gives period that corresponds to an input equillibrium temp
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         teff - Effective Temperature [K]
         teq  - Equillibrium Temperature [K]
         f - Redistribution parameter [1 or 2]
         alb - Albedo 
       OUTPUT:
         teqper - Period of orbit for teq [day]
    """
    # Convert logg and rstar into stellar mass
    mstar = mstar_from_stellarprops(rstar, logg)
    # Calculate semi-major axis [AU] to reach teq
    r_sun = 6.9598e10  #cm
    au2cm = 1.49598e13 # 1 AU = 1.49598e13 cm
    conv = r_sun / au2cm
    semia = 0.5 * (teff/teq)**2  * rstar * np.sqrt(f*(1-alb)) * conv
    # Calculate period of orbit [day]
    teqper = ( semia / (mstar**(1.0/3.0)) )**(3.0/2.0) * 365.25
    return teqper

def teq_from_period(rstar, logg, teff, period, f, alb):
    """Gives equillibrium temperature that corresponds to an input period
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         teff - Effective Temperature [K]
         period  - Period of orbit [day]
         f - Redistribution parameter [1 or 2]
         alb - Albedo 
       OUTPUT:
         teq - Equillibrium temperature [K]
    """
    # Convert logg and rstar into stellar mass
    mstar = mstar_from_stellarprops(rstar, logg)
    # Semi-major axis in AU
    semia = mstar**(1.0/3.0) * (period/365.25)**(2.0/3.0)

    # Calculate Teq [K]
    r_sun = 6.9598e10  #cm
    au2cm = 1.49598e13 # 1 AU = 1.49598e13 cm
    conv = r_sun / au2cm
    teq = teff * np.sqrt(rstar*conv/2.0/semia) * (f*(1.0-alb))**0.25;
    return teq

def prob_to_transit(rstar, logg, per, ecc):
    """Provides probability to transit for fixed eccentricity.
        assumes uniform distribution on cos(inc) orbits.
        ecc is forced to be < 0.99
       INPUT:
         rstar - Radius of star [Rsun]
         logg - log surface gravity [cgs]
         per - Period of orbit [day]
         ecc - Eccentricity 
       OUTPUT:
         prob - probability to transit
    """
    # Replace large ecc values with 0.99
    ecc = np.where(ecc > 0.99, 0.99, ecc)
    # Convert logg and rstar into stellar mass
    mstar = mstar_from_stellarprops(rstar, logg)
    # Semi-major axis in AU
    semia = mstar**(1.0/3.0) * (per/365.25)**(2.0/3.0)
    # probability to transit e=0 including pi/4 effect of cos(inc) dist
    r_sun = 6.9598e10 # cm
    au2cm = 1.49598e13 # 1 AU = 1.49598e13 cm
    prob = (rstar*r_sun) / (semia*au2cm)
    # prob to transit e > 0
    prob = prob / (1.0-ecc**2)
    # Check for cases where prob > 1.0
    prob = np.where(prob > 1.0, 1.0, prob)
    return prob




class tps_planet_detection_metrics:
    """Defines a class to store and read the window function and one sigma
       depth function data relating to the tps planet detection
       metrics.  The tps planet detection metrics consist of the
       window function and one-sigma depth function.  See 
       Burke & Catanzarite 2016, Planet Detection Metrics: Window and 
            One-Sigma Depth Functions for Data Release 25 (KSCI-19101-002)
       for a description of the window function and one-sigma depth function.
       
       instantiate tps_planet_detection_metrics with the KIC ID and filePath
       to the window and OSD function fits files.
       The fits files are available for DR25 here

       http://exoplanetarchive.ipac.caltech.edu/bulk_data_download/

       This function assumes the filenames have not been renamed
       from their original names.
       INIT INPUT:
       wanted_kic - [int] KIC id of target you want data for
       filePath - [str] path to the directory containing fits files
       want_wf - [bool] retrieve window function data
       want_osd - [bool] retrieve one sigma depth function

       CLASS VARIABLE CONTENTS:
       pulsedurations - [hr] list of transit durations searched
       id - [int] Target identifier 
       wf_data - list of dictionary containing window function data
       osd_data - list of dictionary containing one sigma depth function
       
       CLASS FUNCTION CONTENTS:
       __init__ - initialization sets variable contents and reads in files
    """
    def __init__(self, wanted_kic, filePath='', want_wf=True, want_osd=True):
        self.id = 0
        self.wf_data = []
        self.osd_data = []
        windowfunc_suffix = '_dr25_window.fits'
        onesigdepthfunc_suffix = '_dr25_onesigdepth.fits'
        # Force wanted_kic to be int32
        wanted_kic = np.int32(wanted_kic)
        self.id = wanted_kic
        # Get the window function data
        if want_wf:
            windowfunc_filename = os.path.join(filePath,'kplr' + \
                                  '{:09d}'.format(wanted_kic) + \
                                  windowfunc_suffix)
            hdulist_wf = fits.open(windowfunc_filename,mode='readonly')
            for i in range(1,15):
                wfd = {}
                wfd["period"] = np.array(hdulist_wf[i].data["PERIOD"])
                wfd["window"] = np.array(hdulist_wf[i].data["WINFUNC"])
                self.wf_data.append(wfd)
            hdulist_wf.close()

        # Get the one sigma depth function
        if want_osd:
            onesigdepthfunc_filename = os.path.join(filePath,'kplr' + \
                                       '{:09d}'.format(wanted_kic) + \
                                       onesigdepthfunc_suffix)
            hdulist_osd = fits.open(onesigdepthfunc_filename,mode='readonly')
            for i in range(1,15):
                osd = {}
                osd["period"] = np.array(hdulist_osd[i].data["PERIOD"])
                osd["onesigdep"] = np.array(hdulist_osd[i].data["ONESIGDEP"])
                self.osd_data.append(osd)
            hdulist_osd.close()
    


class kepler_detection_efficiency_model_data:
    """ Define a class that contains all the data needed to calculate
        a detection efficiency (DE) model
        The contents of this class can largely be treated as a black box
        as they are filled in by the init function and are very dependent
        on the data tables that are used to build the per-target
        DE model.  More information on this model is available in
        Burke, C.J. & Catanzarite, J. 2017, "Planet Detection Metrics: 
           Per-Target Detection Contours for Data Release 25", KSCI-19111-001

        INIT INPUT:
        filePath - [str] path to the directory containing fits files

        CLASS VARIABLE CONTENTS:
        lowPers, hghPers, midpers, mesValues, fitOrder, fitCoeffs
           These get set by the contents of detectEffData_alpha12_02272017.h5
           These are associated with the period dependent coeffcients for the DE model
        baseDetFunc 
           This is a function that interpolates the Rstar dependent 'base' DE model
           as stored in detectEffData_alpha_base_02272017.txt
        rStarRange - Valid range over Rstar for Rstar 'base' DE model
        cSLSSDetFunc
           This is a function that interpolates the CDPP slope plane DE corrections
           as stored in detectEffData_alpha12_SlopeLongShort_02272017.txt
        cdppSlopeLongRange/cdppSlopeShortRange
           Valid range of the short and long cdpp slope parameters.
        detEffValues - Tabulated DE values for target dependent DE model
        detEffFunctions - Interpolation function for non-smeared DE model
        mesSmearA, mesSmearB - Beta distribution function parameters for MES smearing
        detEffSmearFunctions - Final DE model after MES Smearing applied

        CLASS FUNCTION CONTENTS:
        __init__ - initialization and read in all necessary tables and make
                   DE model interpolating functions
        generate_detection_eff_grid - After instantiation call this function
                to tailor the DE model for the targets parameters
        mes_smearing_parameters - Get the beta distribution parameters
                for the mes smearing model
        final_detEffs - This is the function the end user will actually
                use in order to get a DE model for any arbitary MES and period
    """
    def __init__(self, filePath=''):
        self.lowPers = np.array([0.0])
        self.hghPers = np.array([0.0])
        self.midPers = np.array([0.0])
        self.mesValues = np.array([0.0])
        self.fitOrder = np.array([0.0])
        self.fitCoeffs = np.array([0.0])
        self.baseDetFunc = []
        self.rStarRange = np.array([0.0])
        self.cSLSSDetFunc = []
        self.cdppSlopeLongRange = np.array([0.0])
        self.cdppSlopeShortRange = np.array([0.0])
        self.detEffValues = np.array([0.0])
        self.detEffFunctions = []
        self.mesSmearA = 2.6
        self.mesSmearB = 0.5
        self.detEffSmearFunctions =[]

        # Read in the base Rstar DE model table
        dataBlock = np.genfromtxt(os.path.join(filePath,'detectEffData_alpha_base_02272017.txt'), \
                                    comments='#')
        baseMes = dataBlock[:,0].ravel()
        baseRstar = dataBlock[:,1].ravel()
        baseData = dataBlock[:,2].ravel()
        nMes = 27     
        nRs = 11
        baseMes = np.reshape(baseMes, (nMes,nRs))
        baseRstar = np.reshape(baseRstar, (nMes, nRs))
        baseData = np.reshape(baseData, (nMes, nRs))
        xMes = baseMes[:,0]
        yRs = baseRstar[0,:]
        # Set the interpolating function to interpolate in this table at arbitrary
        #   Mes and Rstar
        self.baseDetFunc = interp.RectBivariateSpline(xMes, yRs, baseData, bbox=[2.5, 30.0, 0.1, 1.25], kx=1, ky=1)
        self.rStarRange = np.array([0.1, 1.25])
        
        # Read in the CDPP Slope plane correction for DE model
        dataBlock = np.genfromtxt(os.path.join(filePath,'detectEffData_alpha12_SlopeLongShort_02272017.txt'), \
                                    comments='#')
        baseSLMes = dataBlock[:,0].ravel()
        baseSL = dataBlock[:,1].ravel()
        baseSS = dataBlock[:,2].ravel()
        baseSLSSData = dataBlock[:,3].ravel()
        nMes = 27     
        nSL = 10
        nSS = 13
        baseSLMes = np.reshape(baseSLMes, (nMes,nSL,nSS))
        baseSL = np.reshape(baseSL, (nMes, nSL,nSS))
        baseSS = np.reshape(baseSS, (nMes, nSL,nSS))
        baseSLSSData = np.reshape(baseSLSSData, (nMes, nSL, nSS))
        xMes = baseSLMes[:,0,0]
        ySL = baseSL[0,:,0]
        zSS = baseSS[0,0,:]
        # Set the interpolating function to interpolate this table at arbitrary
        #  Mes, CdppSlope Long, and CdppSlope Short
        self.cSLSSDetFunc = interp.RegularGridInterpolator((xMes, ySL, zSS), baseSLSSData, bounds_error=False, method='linear', fill_value=None)
        self.cdppSlopeLongRange = np.array([-0.55, 0.4])
        self.cdppSlopeShortRange = np.array([-0.8, 1.2])
       
        # Load DE model coefficients for the final period dependence step.
        fh = h5py.File(os.path.join(filePath,'detectEffData_alpha12_02272017.h5'), 'r')
        self.lowPers = np.array(fh.get('lowPers'))
        self.hghPers = np.array(fh.get('hghPers'))
        self.midPers = np.array(fh.get('midPers'))
        self.mesValues = np.array(fh.get('mesValues'))
        self.fitOrder = np.array(fh.get('fitOrder'))
        self.fitCoeffs = np.array(fh.get('fitCoeffs'))
        
    def generate_detection_eff_grid(self, rstar, cSlopeL, cSlopeS, dutyCycle, \
            dataSpan):
        """ Call after instantiation of class in order to tailor the DE model
            for the target and apply MES smearing to DE model
            INPUT:
            rstar - [Rsun] Target Rstar
            cSlopeL - [float] Target CDPP Slope at long transit duration
            cSlopeS - [float] " short transit duration
            dutyCycle - [float] Duty cycle of valid observations
            dataSpan - [float] Data Span of valid observations
            OUTPUT:
               sets the class contents for MES Smeared DE model        
        """
        nPeriods = len(self.lowPers)
        nValues = len(self.mesValues)
        self.detEffValues = np.zeros((nPeriods, nValues))
        # Get Base Detection efficiency model dependent on rstar
        # NOTE:: Due to oddities in the interp.RectBivariateSpline routine
        #    mes Values MUST BE IN SORTED ORDER
        curMes = self.mesValues
        # Force rstar to be at valid range limits if outside range
        useRstar = self.rStarRange[1] if rstar > self.rStarRange[1] else rstar
        useRstar = self.rStarRange[0] if rstar < self.rStarRange[0] else rstar
        detModel = self.baseDetFunc(curMes, useRstar).ravel()
        # Check CDPP slope parameters are within range
        useCSL = self.cdppSlopeLongRange[1] if cSlopeL > self.cdppSlopeLongRange[1] else cSlopeL
        useCSL = self.cdppSlopeLongRange[0] if cSlopeL < self.cdppSlopeLongRange[0] else cSlopeL
        useCSS = self.cdppSlopeShortRange[1] if cSlopeS > self.cdppSlopeShortRange[1] else cSlopeS
        useCSS = self.cdppSlopeShortRange[0] if cSlopeS < self.cdppSlopeShortRange[0] else cSlopeS
        useCSL = np.full_like(curMes, useCSL)
        useCSS = np.full_like(curMes, useCSS)
        n=len(curMes)
        useCSL = np.reshape(useCSL, (1, n))
        useCSS = np.reshape(useCSS, (1, n))
        curMesArr = np.reshape(curMes, (1,n))
        wantArgs = np.concatenate((curMesArr, useCSL, useCSS), axis=0).T
        # Get the CDPP Slope plane corrections
        detModelSLSS = self.cSLSSDetFunc(wantArgs).ravel()

        # Now make Period dependent DE model
        # There is a DE model for the 5 period bins
        self.detEffFunctions = []
        self.detEffSmearFunctions = []
        # Get the beta distribution mes smearing parameters
        self = self.mes_smearing_parameters(cSlopeL)
        # Get a random set of beta distribution samples
        mesSim = stat.beta.rvs(self.mesSmearA, self.mesSmearB, size=10000)
        # ***Trimming off mesSim < 0.2 for better agreement with 
        # transit injection.  In practice this should be turned off
        # The injections are only up to b<1 which is actually
        # not fully grazing b=(1+rp/rstar) is the actual upper limit***
        # LEAVE FOLLOWING TWO LINES COMMENTED OUT IN PRACTICE
        #idxUse = np.where(mesSim > 0.2)[0]
        #mesSim = mesSim[idxUse]
        # ***If you want to turn off Mes Smearing then use the next line
        # otherwise leave it commented out
        #mesSim = self.mesSmearA / (self.mesSmearA + self.mesSmearB)
        useNtran = dataSpan / self.midPers * dutyCycle
        for ii in range(nPeriods):
            for jj in range(nValues):
                if self.fitOrder[ii, jj] == 1:
                    self.detEffValues[ii, jj] = self.fitCoeffs[ii, jj, 2] + \
                            detModel[jj]
                else:
                    cfs = self.fitCoeffs[ii, jj, :]
                    self.detEffValues[ii, jj] = cfs[0]*useNtran[ii] + \
                        cfs[1]*detModelSLSS[jj] + \
                        cfs[2] + \
                        detModel[jj]
            # Express the detection efficiency as an interpolating function
            useMesValues = np.concatenate(([0.0], self.mesValues))
            useDetEffValues = np.concatenate(([0.0], self.detEffValues[ii,:]))
            # Ensure DE model is in range 0-1
            useDetEffValues = np.where(useDetEffValues < 0.0, 0.0, useDetEffValues)
            useDetEffValues= np.where(useDetEffValues > 1.0, 1.0, useDetEffValues)
            # Define the DE model for this period bin
            detEffFunc = interp.interp1d(useMesValues, useDetEffValues, \
                                        kind='linear', \
                                        copy=False, fill_value=1.0, \
                                        bounds_error=False)
            # Append the DE model for this period bin to class storage detEffFunctions
            self.detEffFunctions.append(detEffFunc)
            # Smear DE model
            useSmearDetEffValues = np.zeros_like(useDetEffValues)
            for jj in range(len(useMesValues)):
                mesRan = mesSim * useMesValues[jj]
                useSmearDetEffValues[jj] = np.mean(detEffFunc(mesRan))
            # Determine the smeared De model for this period bin and append it
            detEffSmearFunc = interp.interp1d(useMesValues, useSmearDetEffValues, \
                                        kind='linear', \
                                        copy=False, fill_value=1.0, \
                                        bounds_error=False)
            self.detEffSmearFunctions.append(detEffSmearFunc)
        return self

    def mes_smearing_parameters(self, CSL):
        """ Define the beta distribution parameters for MES smearing
            INPUT:
            CSL - CDPP Slope Long transit duration
        """
        self.mesSmearA = 5.32468744*CSL*CSL + 3.62297627*CSL + 2.75368232
        self.mesSmearB = -0.28824989*CSL*CSL - 0.11846282*CSL + 0.45067858
        return self

    def final_detEffs(self, mes, per):
        """ This is the function for the end-user to use to get at the DE model
            call this after initilizing class and generate_detection_eff_grid()
            has defined the DE model
            INPUT:
            mes - [float numpy array sigma] - Requested MES for DE model
            per - [float numpy array day] - Requested period for DE model
            OUTPUT: 
            DE model in same shape as mes input array
        """
        newmes = mes.ravel()
        newper = per.ravel()
        newzz = np.zeros_like(newmes)
        # digitize the input periods to the 5 period DE model bins
        periodbin = np.digitize(newper, np.append(self.lowPers, 2000)) - 1
        periodbin = np.where(periodbin < 0, 0, periodbin)
        # Get the DE model for each period bin
        for i in range(len(self.lowPers)):
            idx = np.where(periodbin == i)[0]
            newzz[idx] = self.detEffSmearFunctions[i](newmes[idx])
        # Ensure DE model is well behaved
        newzz = np.where(newzz < 0.0, 0.0, newzz)
        newzz = np.where(newzz > 1.0, 1.0, newzz)
        return np.reshape(newzz, mes.shape)
                


class kepler_single_comp_data:
    """Define a class that contains all the data needed to calculate
       a single target pipeline completeness grid using
       kepler_single_comp()
       CONTENTS:
       id - [int] Target identifier recommend KIC
       period_want - [day] list of orbital periods
       rp_want - [Rearth] list of planet radii
       rstar - [Rsun] star radius
       logg - [cgs] star surface gravity
       teff - [K] stellar effective temperature
       ecc - [0.0 - 1.0] orbital eccentricity
       dataspan - [day] scalar observing baseline duration
       dutycycle - [0.0 -1.0] scalar valid data fraction over dataspan
       pulsedurations - [hr] list of transit durations searched
       cdppSlopeLong - [float] rmsCDPP Slope for long durations
       cdppSlopeShort - [float] rmsCDPP Slope for short durations
       planet_detection_metric_path - [string] directory path
                                        of fits files for the
                                        planet detection metrics
    """
    def __init__(self):
        self.id = 0 
        self.period_want = np.array([0.0])
        self.rp_want = np.array([0.0])
        self.rstar = 0.0
        self.logg = 0.0
        self.teff = 0.0
        self.limbcoeffs = np.array([0.6])
        self.ecc = 0.0
        self.dataspan = 0.0
        self.dutycycle = 0.0
        self.pulsedurations = np.array([0.0])
        self.cdppSlopeLong = -0.5
        self.cdppSlopeShort = -0.5
        self.planet_detection_metric_path = ''


def kepler_single_comp_dr25(data):
    """Calculate a 2D grid of pipeline completeness
       for a single Kepler target.  This is for the DR25 Kepler pipeline only
       INPUT:
         data - instance of class kepler_single_comp_data
       OUTPUT:
         probdet - 2D numpy array of period_want vs rp_want
                   pipeline completeness for single target
         probtot - same as probdet, but includes probability to transit
         DEMod - class kepler_detection_efficiency_model_data that was used
    """
    pulsedurations = np.array([1.5,2.0,2.5,3.0,3.5,4.5,5.0,
                               6.0,7.5,9.0,10.5,12.0,12.5,15.0])
    # Get the planet detection metrics struct
    plan_det_met = tps_planet_detection_metrics(data.id, \
                         data.planet_detection_metric_path, \
                         want_wf=True,want_osd=True)
        
    # Calculate transit duration along period_want list
    transit_duration_1d = transit_duration_zero(data.rstar,
                                           data.logg,
                                           data.period_want,
                                           data.ecc)
    # To avoid extrapolation force the transit duration to be
    #  within the allowed range
    minduration = pulsedurations.min()
    maxduration = pulsedurations.max() - 0.01 
    transit_duration_1d = np.where(transit_duration_1d > maxduration,
                                   maxduration, transit_duration_1d)
    transit_duration_1d = np.where(transit_duration_1d < minduration,
                                   minduration, transit_duration_1d)
    # Now go through transit_duration_1d and create an index array
    # into pulsedurations such that the index points to nearest
    #  pulse without going over
    pulse_index_1d = np.digitize(transit_duration_1d, pulsedurations)
    pulse_index_1d = np.where(pulse_index_1d > 0, pulse_index_1d - 1, pulse_index_1d)
    # Initialize one sigma depth and window function
    one_sigma_depth_1d = np.zeros_like(data.period_want)
    windowfunc_1d = np.zeros_like(data.period_want)
    # iterate over the pulse durations that are present
    for i in range(pulse_index_1d.min(),pulse_index_1d.max()+1):
        idxin = np.arange(data.period_want.size)\
                         [np.nonzero(pulse_index_1d == i)]
        current_period = data.period_want[idxin]
        current_trandur = transit_duration_1d[idxin]
        low_trandur = np.full_like(current_trandur, pulsedurations[i])
        hgh_trandur = np.full_like(current_trandur, pulsedurations[i+1])
        # Start doing one sigma depth function
        low_osd_x = plan_det_met.osd_data[i]['period']
        low_osd_y = plan_det_met.osd_data[i]['onesigdep']
        # Catch inf in osd data
        idxbd = np.where(np.logical_not(np.isfinite(low_osd_y)))[0]
        low_osd_y[idxbd] = 1.0e5
        low_osd_func = interp.interp1d(low_osd_x, low_osd_y,
                                  kind='nearest', copy=False,
                                  assume_sorted=True)
        tmp_period = np.copy(current_period)
        lowper = low_osd_x[0]
        hghper = low_osd_x[-1]
        tmp_period = np.where(tmp_period < lowper, lowper, tmp_period)
        tmp_period = np.where(tmp_period > hghper, hghper, tmp_period)
        current_low_osd = low_osd_func(tmp_period)
        hgh_osd_x = plan_det_met.osd_data[i+1]['period']
        hgh_osd_y = plan_det_met.osd_data[i+1]['onesigdep']
        # Catch inf in osd data
        idxbd = np.where(np.logical_not(np.isfinite(hgh_osd_y)))[0]
        hgh_osd_y[idxbd] = 1.0e5
        hgh_osd_func = interp.interp1d(hgh_osd_x, hgh_osd_y,
                                  kind='nearest', copy=False,
                                  assume_sorted=True)
        tmp_period = np.copy(current_period)
        lowper = hgh_osd_x[0]
        hghper = hgh_osd_x[-1]
        tmp_period = np.where(tmp_period < lowper, lowper, tmp_period)
        tmp_period = np.where(tmp_period > hghper, hghper, tmp_period)
        current_hgh_osd = hgh_osd_func(tmp_period)
        # linear interpolate across pulse durations
        keep_vector = (current_trandur - low_trandur) /\
                      (hgh_trandur - low_trandur)
        current_osd = (current_hgh_osd-current_low_osd) * \
                      keep_vector + current_low_osd
        one_sigma_depth_1d[idxin] = current_osd

        # Start doing window function
        low_wf_x = plan_det_met.wf_data[i]['period']
        low_wf_y = plan_det_met.wf_data[i]['window']
        low_wf_func = interp.interp1d(low_wf_x, low_wf_y,
                                  kind='nearest', copy=False,
                                  assume_sorted=True)
        tmp_period = np.copy(current_period)
        lowper = low_wf_x[0]
        hghper = low_wf_x[-1]
        tmp_period = np.where(tmp_period < lowper, lowper, tmp_period)
        tmp_period = np.where(tmp_period > hghper, hghper, tmp_period)
        current_low_wf = low_wf_func(tmp_period)
        hgh_wf_x = plan_det_met.wf_data[i+1]['period']
        hgh_wf_y = plan_det_met.wf_data[i+1]['window']
        hgh_wf_func = interp.interp1d(hgh_wf_x, hgh_wf_y,
                                  kind='nearest', copy=False,
                                  assume_sorted=True)
        tmp_period = np.copy(current_period)
        lowper = hgh_wf_x[0]
        hghper = hgh_wf_x[-1]
        tmp_period = np.where(tmp_period < lowper, lowper, tmp_period)
        tmp_period = np.where(tmp_period > hghper, hghper, tmp_period)
        current_hgh_wf = hgh_wf_func(tmp_period)
        # linear interpolate across pulse durations
        current_wf = (current_hgh_wf-current_low_wf) * \
                      keep_vector + current_low_wf
        windowfunc_1d[idxin] = current_wf


    # get geometric probability to transit along period_want list
    probtransit_1d = prob_to_transit(data.rstar,data.logg,
                                     data.period_want,data.ecc)
    # Calculate transit depth along rp_want list
    depth_1d = midpoint_transit_depth(data.rstar, data.rp_want, \
                        data.limbcoeffs)

    # Now ready to make things 2d
    nper = data.period_want.size
    nrp = data.rp_want.size
    depth_2d = np.tile(np.reshape(depth_1d,(nrp,1)),nper)
    one_sigma_depth_2d = np.tile(np.reshape(one_sigma_depth_1d,
                                 (1,nper)),(nrp,1))
    period_2d = np.tile(np.reshape(data.period_want,(1,nper)),(nrp,1))
    windowfunc_2d = np.tile(np.reshape(windowfunc_1d,(1,nper)),(nrp,1))
    probtransit_2d = np.tile(np.reshape(probtransit_1d,(1,nper)),(nrp,1))


    # Instantiate detection efficiency object
    #  HINT: If you plan on using KeplerPORTs to make many Det contours
    #  DEMod should be instantiated once outside of module and passed to this function
    #  Then call DEMod.generate_detection_eff_grid only here to tailor it
    #  for the target.  Instantiating the kepler_detection_efficiency_model_data
    #    for every target is a waste of time.
    DEMod = kepler_detection_efficiency_model_data()
    # Generate Detection Efficiency Function tailored for target
    DEMod = DEMod.generate_detection_eff_grid(data.rstar, data.cdppSlopeLong, \
                data.cdppSlopeShort, data.dutycycle, data.dataspan)

    # Do last calculations
    # mes_cor = 1.003 used for better agreement with mes approximation
    #  and mes calculated by TPS
    mes_cor = 1.003
    mes_2d = depth_2d / one_sigma_depth_2d * mes_cor
    zz_2d = DEMod.final_detEffs(mes_2d, period_2d)

    probdet = zz_2d * windowfunc_2d
    probtot = probdet * probtransit_2d
    return probdet, probtot, DEMod

def setup_figure():
    """ Set things for making figures"""
    # Define colors, font sizes, line widths, and marker sizes
    myblack = tuple(np.array([0.0, 0.0, 0.0]) / 255.0)
    mynearblack = tuple(np.array([75.0, 75.0, 75.0]) / 255.0)
    myblue = tuple(np.array([0.0, 109.0, 219.0]) / 255.0)
    myred = tuple(np.array([146.0, 0.0, 0.0]) / 255.0)
    myorange = tuple(np.array([219.0, 209.0, 0.0]) / 255.0)
    myskyblue = tuple(np.array([182.0, 219.0, 255.0]) / 255.0)
    myyellow = tuple(np.array([255.0, 255.0, 109.0]) / 255.0)
    mypink = tuple(np.array([255.0, 182.0, 119.0]) / 255.0)
    labelfontsize = 26.0
    tickfontsize = 26.0
    datalinewidth = 3.0
    plotboxlinewidth = 4.0
    markersize = 1.0
    bkgcolor = 'white'
    axiscolor = myblack
    labelcolor = myblack
    fig = plt.figure(figsize=(8,6), facecolor=bkgcolor)
    ax = plt.gca()
    figstydict={'labelfontsize':labelfontsize, 'tickfontsize':tickfontsize, \
                'datalinewidth':datalinewidth, 'plotboxlinewidth':plotboxlinewidth, \
                'markersize':markersize, 'bkgcolor':bkgcolor, \
                'axiscolor':axiscolor, 'labelcolor':labelcolor, \
                'myblack':myblack, 'mynearblack':mynearblack, \
                'myblue':myblue, 'myred':myred, 'myorange':myorange, \
                'myskyblue':myskyblue, 'myyellow':myyellow, 'mypink':mypink}
    return fig, ax, figstydict

if __name__ == "__main__":
    # Do a few examples of using the KeplerPORTs module
    # Define the detection contour grid points of orbital period and 
    #  planet radius
    min_period = 20.0
    max_period = 730.0
    n_period = 3000
    min_rp = 0.5
    max_rp = 15.0
    n_rp = 2000
    period_want = np.linspace(min_period, max_period, n_period)
    rp_want = np.linspace(np.log10(min_rp), np.log10(max_rp), n_rp)
    period_want_orig = np.copy(period_want)
    rp_want_orig = 10.0**np.copy(rp_want)
    # Define the stellar and noise properties needed for detection contour
    # Begin by making an instance of the class that holds the properties
    #  of detection contour
    # Parameters available from the DR25 Stellar and Occurrence product
    #   table hosted at NASA Exoplanet Archive
    doit = kepler_single_comp_data()
    doit.id = 3429335
    doit.rstar = 0.798
    doit.logg = 4.578
    doit.teff = 5554.0
    doit.dataspan = 1458.93
    doit.dutycycle = 0.874
    doit.limbcoeffs = np.array([0.4869,0.05340,0.5129,-0.3007])
    doit.cdppSlopeLong = -0.4564
    doit.cdppSlopeShort = -0.7051

    # Define detection grid and path to the window function and one-sigma depth
    #  function tables
    doit.period_want = period_want_orig
    doit.rp_want = rp_want_orig
    doit.ecc = 0.0
    doit.planet_detection_metric_path = ''
    # All parameters are set, generate detection contour and detection efficiency
    #  curve
    probdet, probtot, DEMod = kepler_single_comp_dr25(doit)

    # Make figure of Detection Efficiency and its smeared version
    tmpMes = np.linspace(0.0, 30.0, 1000)
    tmpDE = DEMod.detEffFunctions[0](tmpMes)
    tmpSmearDE = DEMod.detEffSmearFunctions[0](tmpMes)
    wantFigure = 'DE_example'
    fig, ax, fsd = setup_figure()
    plt.plot(tmpMes, tmpDE, '--', linewidth=4.0)
    plt.plot(tmpMes, tmpSmearDE, '-', linewidth=4.0)
    plt.xlabel('MES', fontsize=fsd['labelfontsize'], fontweight='heavy')
    plt.ylabel('Detection Efficiency', fontsize=fsd['labelfontsize'], 
               fontweight='heavy')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(fsd['plotboxlinewidth'])
        ax.spines[axis].set_color(fsd['mynearblack'])
    ax.tick_params('both', labelsize=fsd['tickfontsize'], width=fsd['plotboxlinewidth'], 
                   color=fsd['mynearblack'], length=fsd['plotboxlinewidth']*3)
    plt.savefig(wantFigure+'.png',bbox_inches='tight')
    plt.savefig(wantFigure+'.eps',bbox_inches='tight')
    plt.show()
    
    # Make figure of detection contour
    X, Y = np.meshgrid(np.log10(period_want_orig), np.log10(rp_want_orig))
    wantFigure = 'DetContour_example'
    fig, ax, fsd = setup_figure()
    uselevels = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    CS2 = plt.contour(X, Y, probdet, levels=uselevels, linewidth=fsd['datalinewidth'], 
                   colors=(fsd['myblue'],) * len(uselevels))
    plt.clabel(CS2, inline=1, fontsize=fsd['labelfontsize'], fmt='%1.2f', 
           inline_spacing=10.0, fontweight='ultrabold')
    CS1 = plt.contourf(X, Y, probdet, levels=uselevels, cmap=plt.cm.bone)    
    plt.xlabel('Log10(Period) [day]', fontsize=fsd['labelfontsize'], fontweight='heavy')
    plt.ylabel('Log10(R$_{\mathregular{p}}$) [R$_{\mathregular{\oplus}}$]', \
                fontsize=fsd['labelfontsize'], fontweight='heavy')
    ax.set_title('KIC {0:d}'.format(doit.id), fontsize=fsd['labelfontsize']-2)
    ax.title.set_position((0.5,1.03))
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(fsd['plotboxlinewidth'])
        ax.spines[axis].set_color(fsd['mynearblack'])
    ax.tick_params('both', labelsize=fsd['tickfontsize'], width=fsd['plotboxlinewidth'], 
               color=fsd['mynearblack'], length=fsd['plotboxlinewidth']*3)
    plt.savefig(wantFigure+'.png',bbox_inches='tight')
    plt.savefig(wantFigure+'.eps',bbox_inches='tight')
    plt.show()
    
    print("We Will Miss You Kepler!")


