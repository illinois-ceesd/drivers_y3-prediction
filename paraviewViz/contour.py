from paraview.simple import *
import time

def cavityContoursYOH(
  dir,
  iter,
  camera_pos,
  camera_focus,
  textpos = [0.76,0.34], # Defaults to horz. injection location
  rhoux_iso1 = 60.0,
  rhoux_iso2 = 80.0,
  Z_iso = 0.136, # Z_st = 0.136
  YOH_iso = 1e-7,
  #YOH_iso2 = 1e-6,
  #YOH_iso3 = 1e-4,
  #YOH_min = 1e-9,
  #YOH_max = 1e-5,
  prefix = '',
  pixels = [1200,600],
  ):

  # What to plot?
  plot_rhoux = 1
  plot_Z = 1
  plot_YOH = 1

  # Include prefix in output file name, if provided
  if (prefix):
    imageFile= '{}/{}_{:09d}.png'.format(dir,prefix,iter)
  else:
    imageFile= '{}/{:09d}.png'.format(dir,iter)
  solutionFile = '{}/PlasCom2_{:09d}.xdmf'.format(dir,iter)

  # disable automatic camera reset on 'Show'
  paraview.simple._DisableFirstRenderCameraReset()

  #==============================================================================================================
  # Read solution file
  t0 = time.time()

  # Open XDMF and read only subset of data
  data = XDMFReader(FileNames=[solutionFile])
  data.PointArrayStatus = ['O2','C2H4','rho','rhoV-1','imask']
  data.GridStatus = ['grid1','grid6','grid7','grid11']
  #data.GridStatus = ['grid1','grid2','grid3','grid4','grid5','grid6','grid7','grid8','grid9']

  # Get render view and display objects
  renderView = GetActiveViewOrCreate('RenderView')
  mainDisplay = GetDisplayProperties(data)
  mainDisplay.Representation = 'Surface'

  # Color surface white 
  ColorBy(mainDisplay,None)
  mainDisplay.AmbientColor = [1.0, 1.0, 1.0]
  mainDisplay.Opacity = 0.1
  mainDisplay.Specular = 1.0

  readTime = time.time() - t0

  #==============================================================================================================
  # Initial threshold filter for contours
  t0 = time.time()
  threshold = Threshold(Input=data)
  threshold.Scalars = ['POINTS','imask']
  threshold.ThresholdRange = [0.0,1.0]
  Hide(threshold,renderView)

  #============================================================================================================
  if (plot_rhoux):
    # rhoux calculator 
    #rhoux_calculator = Calculator(Input=threshold)
    #rhoux_calculator.ResultArrayName = 'rhoux'
    #rhoux_calculator.Function = 'rhoV-1'
    #rhoux_calculatorDisplay = Show(rhoux_calculator, renderView) # Necessary or else ColorBy(contour) will error
    #Hide(rhoux_calculator,renderView)

    # rhoux contour (freestream)
    rhoux_contour= Contour(Input=threshold) # Comment this when using Calculator
    rhoux_contour.ContourBy = ['POINTS', 'rhoV-1'] # Comment this when using Calculator
    #rhoux_contour = Contour(Input=rhoux_calculator)
    rhoux_contourDisplay = Show(rhoux_contour,renderView)
    rhoux_contourDisplay.Representation = 'Surface'
    rhoux_contour.Isosurfaces = [rhoux_iso1,rhoux_iso2]
    ColorBy(rhoux_contourDisplay, None)
    rhoux_contourDisplay.AmbientColor = [0.4,0.4,1.0]
    rhoux_contourDisplay.DiffuseColor = [0.4,0.4,1.0]
    rhoux_contourDisplay.Opacity = 0.4
    rhoux_contourDisplay.Specular = 1.0

  #============================================================================================================
  if (plot_Z):
    # Z calculator 
    Z_calculator = Calculator(Input=threshold)
    Z_calculator.ResultArrayName = 'Z'
    Z_calculator.Function = '(3.4219*C2H4/rho - O2/rho + 0.54)/(3.4219*1.0 + 0.54)'
    Z_calculatorDisplay = Show(Z_calculator,renderView) # Necessary or else ColorBy(contour) will error
    Hide(Z_calculator,renderView)

    # Z contour 2 (injector)
    Z_contour = Contour(Input=Z_calculator)
    Z_contourDisplay = Show(Z_contour,renderView)
    Z_contourDisplay.Representation = 'Surface'
    Z_contour.Isosurfaces = [Z_iso]
    ColorBy(Z_contourDisplay, None)
    Z_contourDisplay.AmbientColor = [0.0,1.0,0.0]
    Z_contourDisplay.DiffuseColor = [0.0,1.0,0.0]
    Z_contourDisplay.Opacity = 0.05
    Z_contourDisplay.Specular = 1.0

  #============================================================================================================
  # Read in temperature field separately, this time not showing grids and plotting T-contours only
  if (plot_YOH):
    YOH_data = XDMFReader(FileNames=[solutionFile])
    YOH_data.PointArrayStatus = ['imask','OH','rho']
    YOH_data.GridStatus = ['grid1','grid6','grid7','grid11','grid12','grid13'] # Only LIB grids
    renderView = GetActiveViewOrCreate('RenderView')
    YOH_display = GetDisplayProperties(YOH_data)
    YOH_display.Representation = 'Outline'
    YOH_display.Opacity = 0.0

    # Threshold
    threshold_for_YOH = Threshold(Input=YOH_data)
    threshold_for_YOH.Scalars = ['POINTS','imask']
    threshold_for_YOH.ThresholdRange = [0.0,1.0]
    Hide(threshold_for_YOH,renderView)

    # Calculator
    YOH_calculator = Calculator(Input=threshold_for_YOH)
    YOH_calculator.ResultArrayName = 'Y_OH'
    YOH_calculator.Function = 'OH/rho'
    YOH_calculatorDisplay = Show(YOH_calculator,renderView) # Necessary or else ColorBy(contour) will error
    Hide(YOH_calculator,renderView)

    # YOH contours
    YOH_contour = Contour(Input=YOH_calculator)
    YOH_contourDisplay = Show(YOH_contour,renderView)
    YOH_contourDisplay.Representation = 'Surface'
    YOH_contour.Isosurfaces = [YOH_iso]
    ColorBy(YOH_contourDisplay, None)
    YOH_contourDisplay.AmbientColor = [1.0,0.333,0.0]
    YOH_contourDisplay.DiffuseColor = [1.0,0.333,0.0]
    YOH_contourDisplay.Opacity = 1.0
    YOH_contourDisplay.Specular = 1.0

    #YOH_contour = Contour(Input=YOH_calculator)
    #YOH_contour.ContourBy = ['POINTS', 'Y_OH']
    #YOH_contour.Isosurfaces = [YOH_iso1,YOH_iso2,YOH_iso3]
    #YOH_contourDisplay = Show(YOH_contour, renderView)
    #YOH_contourDisplay.Representation = 'Surface'
    #YOH_contourDisplay.Specular = 1.0

    # Transfer functions (log scale)
    #opacity_min = 0.4
    #opacity_max = 1.0
    #YOH_colorTF = GetColorTransferFunction('Y_OH')
    #YOH_opacityTF = GetOpacityTransferFunction('Y_OH')
    #YOH_colorTF.ApplyPreset('Black-Body Radiation', True)
    #YOH_colorTF.RescaleTransferFunction(YOH_min,YOH_max)
    #YOH_opacityTF.RescaleTransferFunction(YOH_min,YOH_max)
    #YOH_colorTF.MapControlPointsToLogSpace()
    #YOH_colorTF.UseLogScale = 1
    #YOH_colorTF.EnableOpacityMapping = 1
    #YOH_opacityTF.Points = [YOH_min,opacity_min,0.5,0.0,YOH_max,opacity_max,0.5,0.0]

  #==============================================================================================================
  # Text labels

  # Oxidizer
  if (plot_rhoux):
    Oxi_text = Text()
    Oxi_text.Text = 'Oxidizer freestream'
    Oxi_textDisplay = Show(Oxi_text,renderView)
    Oxi_textDisplay.FontSize = 12
    Oxi_textDisplay.Color = [0.4,0.4,1.0]
    Oxi_textDisplay.Bold = 1
    Oxi_textDisplay.Shadow = 0
    Oxi_textDisplay.FontFamily = 'Times'
    Oxi_textDisplay.WindowLocation = 'AnyLocation'
    Oxi_textDisplay.Position = [0.1,0.8]
    Oxi_textDisplay = Show(Oxi_text,renderView)

  # Ethylene
  if (plot_Z):
    Ethy_text = Text()
    Ethy_text.Text = """Ethylene\ninjector"""
    Ethy_textDisplay = Show(Ethy_text,renderView)
    Ethy_textDisplay.FontSize = 12
    Ethy_textDisplay.Color = [0.0,1.0,0.0]
    Ethy_textDisplay.Bold = 1
    Ethy_textDisplay.FontFamily = 'Times'
    Ethy_textDisplay.WindowLocation = 'AnyLocation'
    Ethy_textDisplay.Position = textpos
    Ethy_textDisplay = Show(Ethy_text,renderView)

  drawTime = time.time() - t0

  #==============================================================================================================
  # Adjust view, render, and write
  t0 = time.time()

  # Camera, view, lighting
  renderView.CameraPosition = [camera_pos[0],camera_pos[1],camera_pos[2]]
  renderView.CameraFocalPoint = [camera_focus[0],camera_focus[1],0.0]
  renderView.CameraViewAngle = 1.0
  GetRenderView().ViewSize = [pixels[0],pixels[1]]

  # Hide mesh outline and axis
  #mainDisplay.Opacity = 0.0
  renderView.OrientationAxesVisibility = 0 # Hide axes
  renderView.Background = [0.0, 0.0, 0.0] # Black background

  # Render and write to file
  renderView.Update()
  Render()
  WriteImage(imageFile)

  writeTime = time.time() - t0
  print('--> File written: {}.  Read = {:4.1f}s, draw = {:4.1f}s, write = {:4.1f}s'
    .format(imageFile,readTime,drawTime,writeTime))

  #==============================================================================================================
  # Delete data to prevent accumulation between each loop iteration in paraview-driver.py
  Delete(data)
  Delete(mainDisplay)
  Delete(renderView)
  Delete(threshold)

  if (plot_rhoux):
    #Delete(rhoux_calculator)
    Delete(rhoux_contour)
    Delete(rhoux_contourDisplay)

  if (plot_Z):
    Delete(Ethy_text)
    Delete(Ethy_textDisplay)
    Delete(Z_calculator)
    Delete(Z_contour)
    Delete(Z_contourDisplay)

  if (plot_YOH):
    Delete(YOH_data)
    Delete(YOH_display)
    Delete(threshold_for_YOH)
    Delete(YOH_calculator)
    Delete(YOH_contour)
    Delete(YOH_contourDisplay)
    #Delete(YOH_colorTF)
    #Delete(YOH_opacityTF)


def cavityContours(
  dir,
  iter,
  camera_pos,
  camera_focus,
  textpos = [0.76,0.34], # Defaults to horz. injection location
  rhoux_iso1 = 60.0,
  rhoux_iso2 = 80.0,
  Z_iso = 0.136, # Z_st = 0.136
  T_iso1 = 800.0,
  T_iso2 = 1200.0,
  T_iso3 = 2000.0,
  Tmin = 300.0,
  Tmax = 2000.0,
  prefix = '',
  pixels = [1200,600],
  ):

  # What to plot?
  plot_rhoux = 1
  plot_Z = 1
  plot_T = 1

  # Include prefix in output file name, if provided
  if (prefix):
    imageFile= '{}/{}_{:09d}.png'.format(dir,prefix,iter)
  else:
    imageFile= '{}/{:09d}.png'.format(dir,iter)
  solutionFile = '{}/PlasCom2_{:09d}.xdmf'.format(dir,iter)

  # disable automatic camera reset on 'Show'
  paraview.simple._DisableFirstRenderCameraReset()

  #==============================================================================================================
  # Read solution file
  t0 = time.time()

  # Open XDMF and read only subset of data
  data = XDMFReader(FileNames=[solutionFile])
  data.PointArrayStatus = ['O2','C2H4','rho','rhoV-1','imask']
  data.GridStatus = ['grid1','grid6','grid7','grid11']
  #data.GridStatus = ['grid1','grid2','grid3','grid4','grid5','grid6','grid7','grid8','grid9']

  # Get render view and display objects
  renderView = GetActiveViewOrCreate('RenderView')
  mainDisplay = GetDisplayProperties(data)
  mainDisplay.Representation = 'Surface'

  # Color surface white 
  ColorBy(mainDisplay,None)
  mainDisplay.AmbientColor = [1.0, 1.0, 1.0]
  mainDisplay.Opacity = 0.1
  mainDisplay.Specular = 1.0

  readTime = time.time() - t0

  #==============================================================================================================
  # Initial threshold filter for contours
  t0 = time.time()
  threshold = Threshold(Input=data)
  threshold.Scalars = ['POINTS','imask']
  threshold.ThresholdRange = [0.0,1.0]
  Hide(threshold,renderView)

  #============================================================================================================
  if (plot_rhoux):
    # rhoux calculator 
    #rhoux_calculator = Calculator(Input=threshold)
    #rhoux_calculator.ResultArrayName = 'rhoux'
    #rhoux_calculator.Function = 'rhoV-1'
    #rhoux_calculatorDisplay = Show(rhoux_calculator, renderView) # Necessary or else ColorBy(contour) will error
    #Hide(rhoux_calculator,renderView)

    # rhoux contour (freestream)
    rhoux_contour= Contour(Input=threshold) # Comment this when using Calculator
    rhoux_contour.ContourBy = ['POINTS', 'rhoV-1'] # Comment this when using Calculator
    #rhoux_contour = Contour(Input=rhoux_calculator)
    rhoux_contourDisplay = Show(rhoux_contour,renderView)
    rhoux_contourDisplay.Representation = 'Surface'
    rhoux_contour.Isosurfaces = [rhoux_iso1,rhoux_iso2]
    ColorBy(rhoux_contourDisplay, None)
    rhoux_contourDisplay.AmbientColor = [0.4,0.4,1.0]
    rhoux_contourDisplay.DiffuseColor = [0.4,0.4,1.0]
    rhoux_contourDisplay.Opacity = 0.4
    rhoux_contourDisplay.Specular = 1.0

  #============================================================================================================
  if (plot_Z):
    # Z calculator 
    Z_calculator = Calculator(Input=threshold)
    Z_calculator.ResultArrayName = 'Z'
    Z_calculator.Function = '(3.4219*C2H4/rho - O2/rho + 0.54)/(3.4219*1.0 + 0.54)'
    Z_calculatorDisplay = Show(Z_calculator,renderView) # Necessary or else ColorBy(contour) will error
    Hide(Z_calculator,renderView)

    # Z contour 2 (injector)
    Z_contour = Contour(Input=Z_calculator)
    Z_contourDisplay = Show(Z_contour,renderView)
    Z_contourDisplay.Representation = 'Surface'
    Z_contour.Isosurfaces = [Z_iso]
    ColorBy(Z_contourDisplay, None)
    Z_contourDisplay.AmbientColor = [0.0,1.0,0.0]
    Z_contourDisplay.DiffuseColor = [0.0,1.0,0.0]
    Z_contourDisplay.Opacity = 0.05
    Z_contourDisplay.Specular = 1.0

  #============================================================================================================
  # Read in temperature field separately, this time not showing grids and plotting T-contours only
  if (plot_T):
    T_data = XDMFReader(FileNames=[solutionFile])
    T_data.PointArrayStatus = ['imask','temperature']
    T_data.GridStatus = ['grid1','grid6','grid7','grid11','grid12','grid13'] # Only LIB grids
    renderView = GetActiveViewOrCreate('RenderView')
    T_display = GetDisplayProperties(T_data)
    T_display.Representation = 'Outline'
    T_display.Opacity = 0.0

    # Threshold
    threshold_for_T = Threshold(Input=T_data)
    threshold_for_T.Scalars = ['POINTS','imask']
    threshold_for_T.ThresholdRange = [0.0,1.0]
    Hide(threshold_for_T,renderView)

    # T contours
    T_contour = Contour(Input=threshold_for_T)
    T_contour.ContourBy = ['POINTS', 'temperature']
    T_contour.Isosurfaces = [T_iso1,T_iso2,T_iso3]
    T_contourDisplay = Show(T_contour, renderView)
    T_contourDisplay.Representation = 'Surface'

    # Transfer functions (log scale)
    opacity_min = 0.2
    opacity_max = 1.0
    T_colorTF = GetColorTransferFunction('temperature')
    T_opacityTF = GetOpacityTransferFunction('temperature')
    T_colorTF.ApplyPreset('Black-Body Radiation', True)
    T_colorTF.RescaleTransferFunction(Tmin,Tmax)
    T_opacityTF.RescaleTransferFunction(Tmin,Tmax)
    T_colorTF.MapControlPointsToLogSpace()
    T_colorTF.UseLogScale = 1
    T_colorTF.EnableOpacityMapping = 1
    T_opacityTF.Points = [Tmin,opacity_min,0.5,0.0,Tmax,opacity_max,0.5,0.0]

  #==============================================================================================================
  # Text labels

  # Oxidizer
  if (plot_rhoux):
    Oxi_text = Text()
    Oxi_text.Text = 'Oxidizer freestream'
    Oxi_textDisplay = Show(Oxi_text,renderView)
    Oxi_textDisplay.FontSize = 12
    Oxi_textDisplay.Color = [0.4,0.4,1.0]
    Oxi_textDisplay.Bold = 1
    Oxi_textDisplay.Shadow = 0
    Oxi_textDisplay.FontFamily = 'Times'
    Oxi_textDisplay.WindowLocation = 'AnyLocation'
    Oxi_textDisplay.Position = [0.1,0.8]
    Oxi_textDisplay = Show(Oxi_text,renderView)

  # Ethylene
  if (plot_Z):
    Ethy_text = Text()
    Ethy_text.Text = """Ethylene\ninjector"""
    Ethy_textDisplay = Show(Ethy_text,renderView)
    Ethy_textDisplay.FontSize = 12
    Ethy_textDisplay.Color = [0.0,1.0,0.0]
    Ethy_textDisplay.Bold = 1
    Ethy_textDisplay.FontFamily = 'Times'
    Ethy_textDisplay.WindowLocation = 'AnyLocation'
    Ethy_textDisplay.Position = textpos
    Ethy_textDisplay = Show(Ethy_text,renderView)

  drawTime = time.time() - t0

  #==============================================================================================================
  # Adjust view, render, and write
  t0 = time.time()

  # Camera, view, lighting
  renderView.CameraPosition = [camera_pos[0],camera_pos[1],camera_pos[2]]
  renderView.CameraFocalPoint = [camera_focus[0],camera_focus[1],0.0]
  renderView.CameraViewAngle = 1.0
  GetRenderView().ViewSize = [pixels[0],pixels[1]]

  # Hide mesh outline and axis
  #mainDisplay.Opacity = 0.0
  renderView.OrientationAxesVisibility = 0 # Hide axes
  renderView.Background = [0.0, 0.0, 0.0] # Black background

  # Render and write to file
  renderView.Update()
  Render()
  WriteImage(imageFile)

  writeTime = time.time() - t0
  print('--> File written: {}.  Read = {:4.1f}s, draw = {:4.1f}s, write = {:4.1f}s'
    .format(imageFile,readTime,drawTime,writeTime))

  #==============================================================================================================
  # Delete data to prevent accumulation between each loop iteration in paraview-driver.py
  Delete(data)
  Delete(mainDisplay)
  Delete(renderView)
  Delete(threshold)

  if (plot_rhoux):
    #Delete(rhoux_calculator)
    Delete(rhoux_contour)
    Delete(rhoux_contourDisplay)

  if (plot_Z):
    Delete(Ethy_text)
    Delete(Ethy_textDisplay)
    Delete(Z_calculator)
    Delete(Z_contour)
    Delete(Z_contourDisplay)

  if (plot_T):
    Delete(T_contour)
    Delete(T_contourDisplay)
    Delete(T_colorTF)
    Delete(T_opacityTF)

def arcHeaterContours(
  dir,
  iter,
  camera_pos,
  camera_focus,
  YO_iso = 0.001,
  YO2_iso = 0.99,
  YN2_iso = 0.999999999,
  T_iso1 = 4000.0,
  T_iso2 = 10000.0,
  T_iso3 = 16000.0,
  Tmin = 1200.0,
  Tmax = 12000.0,
  opacity = 0.4,
  prefix = '',
  pixels = [1200,600],
  ):

  # Include prefix in output file name, if provided
  if (prefix):
    imageFile= '{}/{}_{:09d}.png'.format(dir,prefix,iter)
  else:
    imageFile= '{}/{:09d}.png'.format(dir,iter)
  solutionFile = '{}/PlasCom2_{:09d}.xdmf'.format(dir,iter)

  # disable automatic camera reset on 'Show'
  paraview.simple._DisableFirstRenderCameraReset()

  #print('camera={}'.format(camera))

  #==============================================================================================================
  # Read solution file
  t0 = time.time()

  # Open XDMF
  data = XDMFReader(FileNames=[solutionFile])

  # Properties modified on data
  data.GridStatus = [\
    'grid1','grid2','grid3','grid4','grid5','grid6','grid7','grid8','grid9','grid10','grid11','grid12','grid13',\
    'grid14','grid15','grid16','grid17','grid18','grid19','grid20','grid21','grid22','grid23','grid24','grid25',\
    'grid26']

  # get active view
  renderView = GetActiveViewOrCreate('RenderView')
  #renderView.AxesGrid.Visibility = 0
  #Hide(data,renderView)

  # Auto-generated
  mainDisplay = GetDisplayProperties(data)
  mainDisplay.Representation = 'Outline'
  materialLibrary = GetMaterialLibrary()
  ColorBy(mainDisplay,('FIELD','vtkBlockColors'))
  mainDisplay.SetScalarBarVisibility(renderView,True)
  vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
  vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')

  # vtkBlockColors --> Solid Color
  ColorBy(mainDisplay,None)
  HideScalarBarIfNotNeeded(vtkBlockColorsLUT,renderView)

  readTime = time.time() - t0

  # What to plot?
  plot_O2 = 1
  plot_N2 = 1
  plot_O = 1
  plot_T = 1
  cut_in_half = 0
  textLabels = 1

  #==============================================================================================================
  # Initial threshold filter for contours
  t0 = time.time()
  threshold = Threshold(Input=data)
  threshold.Scalars = ['POINTS','imask']
  threshold.ThresholdRange = [-1.0,0.0]
  #threshold.ThresholdRange = [-1e20,1e20]
  Hide(threshold,renderView)

  #============================================================================================================
  if (plot_O2):
    # Y_O2 calculator 
    YO2_calculator = Calculator(Input=threshold)
    YO2_calculator.ResultArrayName = 'Y_O2'
    YO2_calculator.Function = 'O2/rho'
    YO2_calculatorDisplay = Show(YO2_calculator, renderView) # Necessary or else ColorBy(contour) will error
    Hide(YO2_calculator,renderView)

    # Y_O2 contour
    YO2_contour = Contour(Input=YO2_calculator)
    YO2_contourDisplay = Show(YO2_contour,renderView)
    YO2_contourDisplay.Representation = 'Surface'
    YO2_contour.Isosurfaces = [YO2_iso]
    ColorBy(YO2_contourDisplay, None)
    YO2_contourDisplay.AmbientColor = [0.33,0.33,1.0]
    YO2_contourDisplay.DiffuseColor = [0.33,0.33,1.0]
    YO2_contourDisplay.Opacity = opacity
    YO2_contourDisplay.Specular = 1.0

  #============================================================================================================
  if (plot_N2):
    # Y_N2 calculator 
    YN2_calculator = Calculator(Input=threshold)
    YN2_calculator.ResultArrayName = 'Y_N2'
    YN2_calculator.Function = '(rho - O - O2 - N - N2O - NO - NO2)/rho'
    YN2_calculatorDisplay = Show(YN2_calculator, renderView) # Necessary or else ColorBy(contour) will error
    Hide(YN2_calculator,renderView)

    # Y_N2 contour
    YN2_contour = Contour(Input=YN2_calculator)
    YN2_contourDisplay = Show(YN2_contour,renderView)
    YN2_contourDisplay.Representation = 'Surface'
    YN2_contour.Isosurfaces = [YN2_iso]
    ColorBy(YN2_contourDisplay, None)
    YN2_contourDisplay.AmbientColor = [0.0,1.0,0.0]
    YN2_contourDisplay.DiffuseColor = [0.0,1.0,0.0]
    YN2_contourDisplay.Opacity = opacity
    YN2_contourDisplay.Specular = 1.0

  #============================================================================================================
  if (plot_O):
    # Y_O calculator 
    YO_calculator = Calculator(Input=threshold)
    YO_calculator.ResultArrayName = 'Y_O'
    YO_calculator.Function = 'O/rho'
    YO_calculatorDisplay = Show(YO_calculator, renderView) # Necessary or else ColorBy(contour) will error
    Hide(YO_calculator,renderView)

    # Y_O contour
    YO_contour = Contour(Input=YO_calculator)
    YO_contourDisplay = Show(YO_contour,renderView)
    YO_contourDisplay.Representation = 'Surface'
    YO_contour.Isosurfaces = [YO_iso]
    ColorBy(YO_contourDisplay, None)
    YO_contourDisplay.AmbientColor = [0.0,1.0,1.0]
    YO_contourDisplay.DiffuseColor = [0.0,1.0,1.0]
    YO_contourDisplay.Opacity = opacity
    YO_contourDisplay.Specular = 1.0

  #============================================================================================================
  if (plot_T == 1):
    # T contours
    T_contour= Contour(Input=threshold)
    T_contour.ContourBy = ['POINTS', 'temperature']
    T_contour.Isosurfaces = [T_iso1,T_iso2,T_iso3]
    T_contourDisplay = Show(T_contour, renderView)
    T_contourDisplay.Representation = 'Surface'

    # Transfer functions (log scale)
    opacity_min = 0.2
    opacity_max = 1.0
    T_colorTF = GetColorTransferFunction('temperature')
    T_opacityTF = GetOpacityTransferFunction('temperature')
    T_colorTF.ApplyPreset('Black-Body Radiation', True)
    T_colorTF.RescaleTransferFunction(Tmin,Tmax)
    T_opacityTF.RescaleTransferFunction(Tmin,Tmax)
    T_colorTF.MapControlPointsToLogSpace()
    T_colorTF.UseLogScale = 1
    T_colorTF.EnableOpacityMapping = 1
    T_opacityTF.Points = [Tmin,opacity_min,0.5,0.0,Tmax,opacity_max,0.5,0.0]

  #==============================================================================================================
  # Final calculator + threshold to show geometry, excluding z>0 points
  Z_calculator = Calculator(Input=threshold)
  Z_calculator.ResultArrayName = 'Z'
  Z_calculator.Function = 'coordsZ'
  Z_calculatorDisplay = Show(Z_calculator,renderView)
  Hide(Z_calculator,renderView)

  Z_threshold = Threshold(Input=Z_calculator)
  Z_thresholdDisplay = Show(Z_threshold,renderView)
  Z_thresholdDisplay.Representation = 'Surface'
  ColorBy(Z_thresholdDisplay, None)
  Z_thresholdDisplay.Opacity = 0.1
  Z_threshold.Scalars = ['POINTS', 'Z']
  if (cut_in_half):
    Z_threshold.ThresholdRange = [-100.0,0.0]
  else:
    Z_threshold.ThresholdRange = [-100.0,100.0]

  #==============================================================================================================
  # Text labels
  if (textLabels == 1):

    # N2
    N2_text = Text()
    N2_text.Text = 'N2 inlets'
    N2_textDisplay = Show(N2_text,renderView)
    N2_textDisplay.FontSize = 12
    N2_textDisplay.Color = [0.4,1.0,0.4]
    N2_textDisplay.Bold = 1
    N2_textDisplay.Shadow = 1
    N2_textDisplay.FontFamily = 'Times'
    N2_textDisplay.WindowLocation = 'AnyLocation'
    N2_textDisplay.Position = [0.19,0.05]
    N2_textDisplay = Show(N2_text,renderView)

    # O2
    O2_text = Text()
    O2_text.Text = 'O2 inlets'
    O2_textDisplay = Show(O2_text,renderView)
    O2_textDisplay.FontSize = 12
    O2_textDisplay.Color = [0.4,0.4,1.0]
    O2_textDisplay.Bold = 1
    O2_textDisplay.FontFamily = 'Times'
    O2_textDisplay.WindowLocation = 'AnyLocation'
    O2_textDisplay.Position = [0.48,0.20]
    O2_textDisplay = Show(O2_text,renderView)

    # O
    O_text = Text()
    O_text.Text = 'O radical'
    O_textDisplay = Show(O_text,renderView)
    O_textDisplay.FontSize = 12
    O_textDisplay.Color = [0.0,1.0,1.0]
    O_textDisplay.Bold = 1
    O_textDisplay.FontFamily = 'Times'
    O_textDisplay.WindowLocation = 'AnyLocation'
    O_textDisplay.Position = [0.73,0.42]
    O_textDisplay = Show(O_text,renderView)

    # Temperature
    T_text = Text()
    T_text.Text = 'Arc heater'
    T_textDisplay = Show(T_text,renderView)
    T_textDisplay.FontSize = 12
    T_textDisplay.Color = [1.0,0.5,0.0]
    T_textDisplay.Bold = 1
    T_textDisplay.FontFamily = 'Times'
    T_textDisplay.WindowLocation = 'AnyLocation'
    T_textDisplay.Position = [0.23,0.58]
    T_textDisplay = Show(T_text,renderView)

  drawTime = time.time() - t0

  #==============================================================================================================
  # Adjust view, render, and write
  t0 = time.time()

  # Camera, view, lighting
  renderView.CameraPosition = [camera_pos[0],camera_pos[1],camera_pos[2]]
  renderView.CameraFocalPoint = [camera_focus[0],camera_focus[1],0.0]
  renderView.CameraViewAngle = 1.0
  GetRenderView().ViewSize = [pixels[0],pixels[1]]

  # Hide mesh outline and axis
  #mainDisplay.Opacity = 0.0
  renderView.OrientationAxesVisibility = 0
  renderView.Background = [0.0, 0.0, 0.0] # Black background

  # Render and write to file
  renderView.Update()
  Render()
  WriteImage(imageFile)

  writeTime = time.time() - t0
  print('--> File written: {}.  Read = {:4.1f}s, draw = {:4.1f}s, write = {:4.1f}s'
    .format(imageFile,readTime,drawTime,writeTime))

  #==============================================================================================================
  # Delete data to prevent accumulation between each loop iteration in paraview-driver.py
  Delete(data)
  Delete(mainDisplay)
  Delete(renderView)
  Delete(materialLibrary)
  Delete(threshold)

  if (plot_O2):
    Delete(YO2_calculator)
    Delete(YO2_contour)
    Delete(YO2_contourDisplay)

  if (plot_N2):
    Delete(YN2_calculator)
    Delete(YN2_contour)
    Delete(YN2_contourDisplay)

  if (plot_O):
    Delete(YO_calculator)
    Delete(YO_contour)
    Delete(YO_contourDisplay)

  if (plot_T):
    Delete(T_contour)
    Delete(T_contourDisplay)
    Delete(T_colorTF)
    Delete(T_opacityTF)

  if (textLabels):
    Delete(N2_text)
    Delete(N2_textDisplay)
    Delete(O2_text)
    Delete(O2_textDisplay)
    Delete(O_text)
    Delete(O_textDisplay)
    Delete(T_text)
    Delete(T_textDisplay)

  Delete(Z_calculator)
  Delete(Z_threshold)
  Delete(Z_thresholdDisplay)
