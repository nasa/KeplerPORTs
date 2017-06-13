# KeplerPORTs
KeplerPORTs.py - Illustrate making use of numerous Kepler Planet Occurrence Rate Data Products for Data Release 25 and SOC 9.3 Kepler Pipeline version.  This code generates a detection contour according to the documentation

Burke, C.J. & Catanzarite, J. 2017, "Planet Detection Metrics: Per-Target Detection Contours for Data Release 25", KSCI-19111-001

Additional recommended background reading

- Earlier Data Release 24 version of detection contour
    * Burke et al. 2015, ApJ, 809, 8
- Transit injection and recovery tests for the Kepler pipeline
    * Christiansen et al. 2013, ApJS, 207, 35
    * Christiansen et al. 2015, ApJ, 810, 95   (One Year Kepler data)
    * Christiansen et al. 2016, ApJ, 828, 99   (Data Release 24)
    * Christiansen, J. L. 2017, Planet Detection Metrics: Pixel-Level Transit Injection Tests of Pipeline Detection Efficiency for Data Release 25 (KSCI-19110-001)
    * Burke & Catanzarite 2017, Planet Detection Metrics: Per-Target Flux-Level Transit Injection Tests of TPS for Data Release 25 (KSCI-19109-001)
- Kepler Target Noise and Data Quality metrics
    * Burke & Catanzarite 2016, Planet Detection Metrics: Window and One-Sigma Depth Functions for Data Release 25 (KSCI-19101-002)
    
**Assumptions** - python packages numpy, scipy, matplotlib, astropy, and h5py are available and files

- detectEffData_alpha12_02272017.h5
- detectEffData_alpha12_SlopeLongShort_02272017.txt
- detectEffData_alpha_base_02272017.txt
- kplr003429335_dr25_onesigdepth.fits
- kplr003429335_dr25_window.fits
are available in the same directory as KeplerPORTs.py

**Running**: python KeplerPORTs.py

**Output**: Displays a series of figures and generates hardcopy

Notices:

Copyright © 2017 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

NASA acknowledges the SETI Institute’s primary role in authoring and producing the KeplerPORTs (Kepler Planet Occurrence Rate Tools) under Cooperative Agreement Number NNX13AD01A.


Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
