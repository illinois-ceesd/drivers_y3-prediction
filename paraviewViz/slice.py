from paraview.simple import *
import time

def simpleSlice(
  dir,
  iter,
  varName,
  varMin,
  varMax,
  camera,
  colorScheme = 'Cool to Warm',
  logScale = 0,
  invert = 0,
  func = '',
  cbTitle = '',
  prefix = '',
  pixels = [1200,600],
  sliceNormal = [0.0,0.0,1.0],
  sliceOrigin = [0.0,0.0,0.0]):

  # Include prefix in output file name, if provided
  if (prefix):
    imageFile= '{}/{}-{}_{:06d}.png'.format(dir,prefix,varName,iter)
  else:
    imageFile= '{}/{}_{:06d}.png'.format(dir,varName,iter)
  solutionFile = '{}/prediction-fluid-{:06d}.pvtu'.format(dir,iter)

  # If func not provided, simply set it to varName -- everything is a Calculator
  if (not func):
    func = varName

  # disable automatic camera reset on 'Show'
  paraview.simple._DisableFirstRenderCameraReset()

  #print('func={},prefix={},camera={},pixels={}'.format(func,prefix,camera,pixels))

  #==============================================================================================================
  # Read solution file
  t0 = time.time()

  # Open XDMF
  #data = XDMFReader(FileNames=[solutionFile])
  data = XMLPartitionedUnstructuredGridReader(FileName=[solutionFile])
  #data.PointArrayStatus = ['cv_mass', 'cv_energy', 'cv_momentum', 'dv_temperature', 'dv_pressure', 'mach', 'velocity', 'sponge_sigma', 'alpha', 'tagged_cells', 'cfl']
  #data.PointArrayStatus = ['cv_mass', 'cv_energy', 'cv_momentum', 'dv_temperature', 'dv_pressure', 'mach', 'velocity', 'sponge_sigma', 'alpha', 'tagged_cells', 'cfl', 'Y_air', 'Y_fuel']
  data.PointArrayStatus = ['cv_mass', 'cv_energy', 'cv_momentum', 'dv_temperature', 'dv_pressure', 'mach', 'velocity', 'sponge_sigma', 'alpha', 'tagged_cells', 'cfl', 'Y_C2H4', 'Y_H2', 'Y_H2O', 'Y_O2', 'Y_CO', 'Y_fuel', 'Y_air', 'mu']
  readTime = time.time() - t0

  # Properties modified on data

  # get active view
  renderView = GetActiveViewOrCreate('RenderView')

  # display properties
  mainDisplay = GetDisplayProperties(data)
  mainDisplay.Representation = 'Outline'
  materialLibrary = GetMaterialLibrary()
  mainDisplay.SetScalarBarVisibility(renderView,True)

  #============================================================================================================
  # Calculator filter
  var = Calculator(Input=data)
  var.ResultArrayName = varName
  var.Function = func
  Delete(data)

  # Gets rid of "Failed to determine the LookupTable being used" error
  calculatorDisplay = Show(var,renderView)
  ColorBy(calculatorDisplay,('POINTS',varName))
  calculatorDisplay.SetScalarBarVisibility(renderView,True)
  #Hide(var,renderView)

  #dataObject = Show(data, renderView, 'UnstructuredGridRepresentation')
  #dataObject.Representation = 'Surface'

  #ColorBy(mainDisplay, ('POINTS', 'mach'))

  # Color scheme settings
  colorTF = GetColorTransferFunction(varName)
  opacityTF = GetOpacityTransferFunction(varName)
  colorTF.ApplyPreset(colorScheme,True)
  colorTF.RescaleTransferFunction(varMin,varMax)
  opacityTF.RescaleTransferFunction(varMin,varMax)

  # Log scale -- comes after rescaling color bar
  if (logScale == 1):
    colorTF.MapControlPointsToLogSpace()
    colorTF.UseLogScale = 1

  # Invert color scheme, if requested
  if (invert == 1):
    colorTF.InvertTransferFunction()

  #==============================================================================================================
  # Colorbar
  colorbar = GetScalarBar(colorTF,renderView)
  colorbar.Title = cbTitle
  colorbar.TitleJustification = 'Centered'
  colorbar.TitleBold = 1
  colorbar.TitleItalic = 0
  colorbar.TitleShadow = 1
  colorbar.TitleFontSize = 20
  colorbar.LabelBold = 1
  colorbar.LabelItalic = 0
  colorbar.LabelShadow = 1
  colorbar.LabelFontSize = 15
  colorbar.ScalarBarLength = 0.33
  colorbar.WindowLocation = 'Upper Center'
  colorbar.Orientation = 'Horizontal'

  # Get rid of default color bar -- must come after main var
  #if (varName != 'rho'):
    #GetColorTransferFunction('rho').RescaleTransferFunction(0.0,1.0)
    #GetOpacityTransferFunction('rho').RescaleTransferFunction(0.0,1.0)
    #HideScalarBarIfNotNeeded(GetColorTransferFunction('rho'),renderView)

  drawTime = time.time() - t0 - readTime

  #==============================================================================================================
  # Adjust view, render, and write

  # Camera, view, lighting
  print(f"camera {camera}")
  renderView.ResetCamera()
  renderView.InteractionMode = '2D'
  renderView.CameraPosition = [camera[0],camera[1],camera[2]]
  renderView.CameraFocalPoint = [camera[0],camera[1],0.0]
  renderView.CameraViewAngle = 1.0
  renderView.CameraParallelScale = camera[2]
  GetRenderView().UseLight = 0
  GetRenderView().ViewSize = [pixels[0],pixels[1]]

  # Hide mesh outline and axis
  #mainDisplay.Opacity = 0.0
  renderView.OrientationAxesVisibility = 0

  # Render and write to file
  renderView.Update()
  Render()
  WriteImage(imageFile)

  writeTime = time.time() - t0 - drawTime - readTime
  print('--> File written: {}.  Read = {:4.1f}s, draw = {:4.1f}s, write = {:4.1f}s'
    .format(imageFile,readTime,drawTime,writeTime))

  #==============================================================================================================
  # Delete data to prevent accumulation between each outer loop iteration
  Delete(var)
  Delete(renderView)
  Delete(mainDisplay)
  #Delete(materialLibrary)
  Delete(colorTF)
  Delete(opacityTF)
  Delete(colorbar)

