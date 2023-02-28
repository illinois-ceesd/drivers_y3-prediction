# To run with MPI, use mpiexec -n 4 pvbatch --mpi --symmetric paraview-driver.py [arguments]

# Import Paraview functions 
from mpi4py import MPI
import sys
import math
from contour import *
from slice import *

# Command-line arguments
dir = sys.argv[1]
prefix = sys.argv[2]
startIterGlobal = int(sys.argv[3])
if (len(sys.argv) == 4):
  skipIter = 1
  stopIterGlobal = startIterGlobal
elif (len(sys.argv) == 6):
  skipIter = int(sys.argv[4])
  stopIterGlobal = int(sys.argv[5])
else:
  print('There are two possible argument lists:')
  print('3 arguments: (1) dir, (2) prefix, (3) iter')
  print('5 arguments: (1) dir, (2) prefix, (3) startIter, (4) skipIter, (5) stopIter.')
  print('Aborting.')
  exit()

# Divide work evenly, splitting remainder
#comm = MPI.COMM_WORLD
#nProcs = comm.Get_size()
#rank = comm.Get_rank()
#nIterGlobal = int( (stopIterGlobal - startIterGlobal)/skipIter + 1 )
#nIter = math.floor(nIterGlobal/nProcs)
#remainder = nIterGlobal % nProcs
#if (rank < remainder):
  #startIter = (nIter+1) * rank
  #stopIter = startIter + nIter
#else:
  #startIter = nIter*rank + remainder
  #stopIter = startIter + (nIter-1)

## Scale by skipIter, shift by startIterGlobal
#nIter = stopIter - startIter + 1
#startIter = startIter*skipIter + startIterGlobal
#stopIter = stopIter*skipIter + startIterGlobal

# If not running with MPI
startIter = startIterGlobal
stopIter = stopIterGlobal

# Announce
#if (rank == 0):
  #print('PARAVIEW DRIVER: Splitting global iter = [{:09d} : {:d} : {:09d}]'.
    #format(startIterGlobal,skipIter,stopIterGlobal))
#comm.barrier()
#for p in range(0,nProcs):
  #if (rank == p):
    #print('  rank={}: nIter = {}, iter = [{:09d} : {:d} : {:09d}]'.
      #format(rank,nIter,startIter,skipIter,stopIter))
  #comm.barrier()

# Loop
#print('PARAVIEW DRIVER: Making images for iter = [{:09d} : {:d} : {:09d}]'.
  #format(startIterGlobal,skipIter,stopIterGlobal))
if (startIter == stopIter):
  iter = [startIter]
else:
  iter = range(startIter,stopIter,skipIter)

for n in iter:

  if (prefix == 'upstream_full'): # Upstream arc-heater, full geometry

    camera = [0.165,0.012,5.0]
    pixels = [1600,400]

    imaskedSlice(dir,n,'rho',0.04,4.0,camera,invert=1,
      colorScheme='erdc_rainbow_dark',prefix=prefix,pixels=pixels,cbTitle='Density [kg/m^3]')

    imaskedSlice(dir,n,'pressure',70e3,340e3,camera,
      colorScheme='GREEN-WHITE_LINEAR',prefix=prefix,pixels=pixels,cbTitle='Pressure [Pa]')

    imaskedSlice(dir,n,'total-pressure',70e3,340e3,camera,
      func='pressure + 0.5*rho*(velocity-1^2 + velocity-2^2 + velocity-3^2)',
      colorScheme='GREEN-WHITE_LINEAR',prefix=prefix,pixels=pixels,cbTitle='Total pressure [Pa]')

    imaskedSlice(dir,n,'temperature',250.0,25000.0,camera,logScale=1,
      colorScheme='Black-Body Radiation',prefix=prefix,pixels=pixels,cbTitle='Temperature [K]')

    imaskedSlice(dir,n,'velocity-1',-800.0,800.0,camera,
      colorScheme='Cool to Warm (Extended)',prefix=prefix,pixels=pixels,cbTitle='X-velocity [m/s]')

    imaskedSlice(dir,n,'speed',0.0,800.0,camera,func='(velocity-1^2 + velocity-2^2 + velocity-3^2)^(0.5)',
      colorScheme='Grayscale',prefix=prefix,pixels=pixels,cbTitle='Velocity magnitude [m/s]')

    imaskedSlice(dir,n,'Y_O',1e-8,1e-3,camera,func='O/rho',logScale=1,
      colorScheme='blue2cyan',prefix=prefix,pixels=pixels,cbTitle='O mass fraction')

    imaskedSlice(dir,n,'Y_N',1e-9,1e-5,camera,func='N/rho',logScale=1,
      colorScheme='erdc_gold_BW',prefix=prefix,pixels=pixels,cbTitle='N mass fraction')

    imaskedSlice(dir,n,'Y_NO',1e-8,1e-2,camera,func='NO/rho',logScale=1,
      colorScheme='Inferno (matplotlib)',prefix=prefix,pixels=pixels,cbTitle='NO mass fraction')

    imaskedSlice(dir,n,'Y_NO2',1e-8,1e-2,camera,func='NO2/rho',logScale=1,
      colorScheme='Inferno (matplotlib)',prefix=prefix,pixels=pixels,cbTitle='NO2 mass fraction')

  elif (prefix == 'N2-injector'): # Upstream injector, grid 11 in paraview
    camera = [0.0025,0.019,0.6]
    pixels = [900,700]

    #imaskedSlice(dir,n,'rho',3.6,4.3,camera,invert=1,
      #colorScheme='erdc_rainbow_dark',prefix=prefix,pixels=pixels,cbTitle='Density [kg/m^3]')

    #imaskedSlice(dir,n,'pressure',300e3,400e3,camera,
      #colorScheme='GREEN-WHITE_LINEAR',prefix=prefix,pixels=pixels,cbTitle='Pressure [Pa]')

    #imaskedSlice(dir,n,'temperature',275.0,300.0,camera,
      #colorScheme='Black-Body Radiation',prefix=prefix,pixels=pixels,cbTitle='Temperature [K]')

    imaskedSlice(dir,n,'rhoumag',0.0,700.0,camera,func='sqrt((rhoV-1)^2+(rhoV-2)^2+(rhoV-3)^2)',
      colorScheme='Rainbow Blended Black',prefix=prefix,pixels=pixels,cbTitle='Momentum magnitude [N s/m^3]')

  elif (prefix == 'arc-heater-3d'): # 3D contours, upstream arc-heater
    camera_pos = [-2,2,3]
    camera_focus = [0.06,-0.002,0]
    pixels = [1100,600]

    arcHeaterContours(dir,n,camera_pos,camera_focus,pixels=pixels,prefix=prefix)

  elif (prefix == 'cavity-horz-3d'): # 3D contours, downstream cavity, horizontal injection
    camera_pos = [-1.8,0.8,3]
    camera_focus = [0.68,-0.005,0]
    pixels = [1000,600]
    textpos = [0.72,0.24]

    prefix2 = prefix+'_T'
    cavityContours(dir,n,camera_pos,camera_focus,textpos,pixels=pixels,prefix=prefix2)
    prefix2 = prefix+'_OH'
    cavityContoursYOH(dir,n,camera_pos,camera_focus,textpos,pixels=pixels,prefix=prefix2)

  elif (prefix == 'cavity-vert-3d'): # 3D contours, downstream cavity, vertical injection
    camera_pos = [-1.8,0.8,3]
    camera_focus = [0.68,-0.005,0]
    pixels = [1000,600]
    textpos = [0.18,0.06]

    prefix2 = prefix+'_T'
    cavityContours(dir,n,camera_pos,camera_focus,textpos,pixels=pixels,prefix=prefix2)
    prefix2 = prefix+'_OH'
    cavityContoursYOH(dir,n,camera_pos,camera_focus,textpos,pixels=pixels,prefix=prefix2)


  elif (prefix == 'cavity_z'): # Downstream cavity

    # View approx x=[0.635,0.735], y=[-0.034,0.026],  cavity is approx x=[0.65,0.72]
    camera = [0.685,-0.004,4.0]
    pixels = [1000,600]

    imaskedSlice(dir,n,'rho',0.04,0.4,camera,invert=1,
      colorScheme='erdc_rainbow_dark',prefix=prefix,pixels=pixels,cbTitle='Density [kg/m^3]')

    imaskedSlice(dir,n,'pressure',3000,60000,camera,
      colorScheme='GREEN-WHITE_LINEAR',prefix=prefix,pixels=pixels,cbTitle='Pressure [Pa]')

    imaskedSlice(dir,n,'temperature',200.0,800.0,camera,
      colorScheme='Black-Body Radiation',prefix=prefix,pixels=pixels,cbTitle='Temperature [K]')

    imaskedSlice(dir,n,'velocity-1',-1400.0,1400.0,camera,
      colorScheme='Cool to Warm (Extended)',prefix=prefix,pixels=pixels,cbTitle='X-velocity [m/s]')

    imaskedSlice(dir,n,'speed',0.0,1400.0,camera,func='(velocity-1^2 + velocity-2^2 + velocity-3^2)^(0.5)',
      colorScheme='bone_Matlab',prefix=prefix,pixels=pixels,cbTitle='Velocity magnitude [m/s]')

    #imaskedSlice(dir,n,'Y_C2H4',1e-3,1.0,camera,func='C2H4/rho',logScale=1,
      #colorScheme='RED-PURPLE',prefix=prefix,pixels=pixels,cbTitle='C2H4 mass fraction')

    imaskedSlice(dir,n,'Z',0.0,0.272,camera,func='(3.4219*C2H4/rho - O2/rho + 0.54)/(3.4219*1.0 + 0.54)',
      colorScheme='GyRd',prefix=prefix,pixels=pixels,cbTitle='Mixture fraction')

  elif (prefix == 'precavity_bl'): # BL before cavity; z_camera=4.0 --> 10mm/100px

    # View approx x=[0.52,0.67], y=[-0.02,0.02], cavity is approx x=[0.65,0.72]
    camera = [0.595,0.0,4.0]
    pixels = [1500,500]

    imaskedSlice(dir,n,'rho',0.06,0.2,camera,invert=1,
      colorScheme='erdc_rainbow_dark',prefix=prefix,pixels=pixels,cbTitle='Density [kg/m^3]')

    imaskedSlice(dir,n,'pressure',5000,30000,camera,
      colorScheme='GREEN-WHITE_LINEAR',prefix=prefix,pixels=pixels,cbTitle='Pressure [Pa]')

    imaskedSlice(dir,n,'temperature',250.0,600.0,camera,
      colorScheme='Black-Body Radiation',prefix=prefix,pixels=pixels,cbTitle='Temperature [K]')

    imaskedSlice(dir,n,'velocity-1',-1400.0,1400.0,camera,
      colorScheme='Cool to Warm (Extended)',prefix=prefix,pixels=pixels,cbTitle='X-velocity [m/s]')

  elif (prefix == 'nozzle_to_cavity'): # Downstream section between nozzle and cavity

    # View approx x=[0.30,0.66], y=[-0.02,0.02], cavity is approx x=[0.65,0.72]
    camera = [0.50,0.0,4.0]
    pixels = [2400,300]

    imaskedSlice(dir,n,'rho',0.06,0.2,camera,invert=1,
      colorScheme='erdc_rainbow_dark',prefix=prefix,pixels=pixels,cbTitle='Density [kg/m^3]')

    imaskedSlice(dir,n,'pressure',5000,30000,camera,
      colorScheme='GREEN-WHITE_LINEAR',prefix=prefix,pixels=pixels,cbTitle='Pressure [Pa]')

    imaskedSlice(dir,n,'temperature',250.0,600.0,camera,
      colorScheme='Black-Body Radiation',prefix=prefix,pixels=pixels,cbTitle='Temperature [K]')

    imaskedSlice(dir,n,'velocity-1',-1400.0,1400.0,camera,
      colorScheme='Cool to Warm (Extended)',prefix=prefix,pixels=pixels,cbTitle='X-velocity [m/s]')

  elif (prefix == 'nozzle_down_z'): # Downstream nozzle

    camera = [0.285,0.004,4.0]
    pixels = [1000,600]

    imaskedSlice(dir,n,'rho',0.02,10.0,camera,invert=1,
      colorScheme='erdc_rainbow_dark',prefix=prefix,pixels=pixels,cbTitle='Density [kg/m^3]')

    imaskedSlice(dir,n,'pressure',3000,9e5,camera,
      colorScheme='GREEN-WHITE_LINEAR',prefix=prefix,pixels=pixels,cbTitle='Pressure [Pa]')

    imaskedSlice(dir,n,'temperature',120.0,1100.0,camera,logScale=1,
      colorScheme='Black-Body Radiation',prefix=prefix,pixels=pixels,cbTitle='Temperature [K]')

    imaskedSlice(dir,n,'velocity-1',-1300.0,1300.0,camera,
      colorScheme='Cool to Warm (Extended)',prefix=prefix,pixels=pixels,cbTitle='X-velocity [m/s]')

  else:
    print('Unrecognized prefix: {}'.format(prefix))
  
# Announce
#print('PARAVIEW DRIVER: Done.\n')
