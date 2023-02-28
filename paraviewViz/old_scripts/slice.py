from paraview.simple import *
import time

def imaskedSlice(
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
    imageFile= '{}/{}-{}_{:09d}.png'.format(dir,prefix,varName,iter)
  else:
    imageFile= '{}/{}_{:09d}.png'.format(dir,varName,iter)
  solutionFile = '{}/PlasCom2_{:09d}.xdmf'.format(dir,iter)

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
  data = XDMFReader(FileNames=[solutionFile])

  # Properties modified on data
  data.GridStatus = [\
    'grid1','grid2','grid3','grid4','grid5','grid6','grid7','grid8','grid9','grid10','grid11','grid12','grid13',\
    'grid14','grid15','grid16','grid17','grid18','grid19','grid20','grid21','grid22','grid23','grid24','grid25',\
    'grid26']

  # get active view
  renderView = GetActiveViewOrCreate('RenderView')

  # Auto-generated
  mainDisplay = GetDisplayProperties(data)
  mainDisplay.Representation = 'Outline'
  #renderView.ResetCamera()
  materialLibrary = GetMaterialLibrary()
  ColorBy(mainDisplay,('FIELD','vtkBlockColors'))
  mainDisplay.SetScalarBarVisibility(renderView,True)
  vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
  vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')

  # vtkBlockColors --> Solid Color
  ColorBy(mainDisplay,None)
  HideScalarBarIfNotNeeded(vtkBlockColorsLUT,renderView)

  readTime = time.time() - t0

  #============================================================================================================
  # Calculator filter
  var = Calculator(Input=data)
  var.ResultArrayName = varName
  var.Function = func
  Delete(data)

  # Gets rid of "Failed to determine the LookupTable being used" error
  calculator1Display = Show(var,renderView)
  ColorBy(calculator1Display,('FIELD','vtkBlockColors'))
  #calculator1Display.SetScalarBarVisibility(renderView,True)
  Hide(var,renderView)

  #============================================================================================================
  # Slice filter
  t0 = time.time()

  # create a new 'Slice'
  slice = Slice(Input=var)

  # show data in view
  sliceDisplay = Show(slice,renderView)
  sliceDisplay.Representation = 'Surface'
  sliceDisplay.SetScalarBarVisibility(renderView,True)

  # Set origin and normal vector of slice
  slice.SliceType.Origin = sliceOrigin
  slice.SliceType.Normal = sliceNormal

  # Misc
  ColorBy(sliceDisplay,('POINTS',varName))
  sliceDisplay.RescaleTransferFunctionToDataRange(True,False)
  sliceDisplay.SetScalarBarVisibility(renderView,True)

  #==============================================================================================================
  # Threshold filter
  threshold = Threshold(Input=slice)
  threshold.Scalars = ['POINTS','imask']
  threshold.ThresholdRange = [0.0,1.0]
  thresholdDisplay = Show(threshold,renderView)
  thresholdDisplay.Representation = 'Surface'
  Hide(slice,renderView) # Hide slice data in view

  # Color scheme settings -- must come after threshold
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
  colorbar.WindowLocation = 'UpperCenter'
  colorbar.Orientation = 'Horizontal'

  # Get rid of default color bar -- must come after main var
  if (varName != 'rho'):
    GetColorTransferFunction('rho').RescaleTransferFunction(0.0,1.0)
    GetOpacityTransferFunction('rho').RescaleTransferFunction(0.0,1.0)
    HideScalarBarIfNotNeeded(GetColorTransferFunction('rho'),renderView)
  
  drawTime = time.time() - t0

  #==============================================================================================================
  # Adjust view, render, and write
  t0 = time.time()

  # Camera, view, lighting
  renderView.CameraPosition = [camera[0],camera[1],camera[2]]
  renderView.CameraFocalPoint = [camera[0],camera[1],0.0]
  renderView.CameraViewAngle = 1.0
  GetRenderView().UseLight = 0
  GetRenderView().ViewSize = [pixels[0],pixels[1]]

  # Hide mesh outline and axis
  mainDisplay.Opacity = 0.0
  renderView.OrientationAxesVisibility = 0

  # Render and write to file
  renderView.Update()
  Render()
  WriteImage(imageFile)

  writeTime = time.time() - t0
  print('--> File written: {}.  Read = {:4.1f}s, draw = {:4.1f}s, write = {:4.1f}s'
    .format(imageFile,readTime,drawTime,writeTime))

  #==============================================================================================================
  # Delete data to prevent accumulation between each outer loop iteration
  Delete(var)
  Delete(threshold)
  Delete(renderView)
  Delete(mainDisplay)
  Delete(thresholdDisplay)
  Delete(materialLibrary)
  Delete(slice)
  Delete(colorTF)
  Delete(opacityTF)
  Delete(colorbar)

def slice(dir,iter,var,varMin,varMax,colorScheme,logScale,invert):

  imageFile= '{}/slice-{}_{:09d}.png'.format(dir,var,iter)
  solutionFile = '{}/PlasCom2_{:09d}.xdmf'.format(dir,iter)

  camera[0] = 0.165
  camera[1] = 0.012
  camera[2] = 5.0
  xpixels = 1600 
  ypixels = 400

  sliceNormal = [0.0,0.0,1.0]
  sliceOrigin = [0.0,0.0,0.0]

  #### disable automatic camera reset on 'Show'
  paraview.simple._DisableFirstRenderCameraReset()

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

  # show data in view
  dataDisplay = GetDisplayProperties(data)

  # trace defaults for the display properties.
  dataDisplay.Representation = 'Outline'

  # reset view to fit data
  renderView.ResetCamera()

  # get the material library
  materialLibrary = GetMaterialLibrary()

  # set scalar coloring
  ColorBy(dataDisplay,('FIELD','vtkBlockColors'))

  # show color bar/color legend
  dataDisplay.SetScalarBarVisibility(renderView,True)

  # get color/opacity transfer function and color/opacity map for 'vtkBlockColors'
  vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
  vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')

  # vtkBlockColors --> Solid Color
  ColorBy(dataDisplay,None)
  HideScalarBarIfNotNeeded(vtkBlockColorsLUT,renderView)

  readTime = time.time() - t0
  #==============================================================================================================
  # Slice
  t0 = time.time()

  # create a new 'Slice'
  slice = Slice(Input=data)

  # show data in view
  sliceDisplay = Show(slice,renderView)
  sliceDisplay.Representation = 'Surface'

  # show color bar/color legend
  sliceDisplay.SetScalarBarVisibility(renderView,True)

  # update the view to ensure updated data information

  # Properties modified on slice.SliceType
  slice.SliceType.Origin = sliceOrigin
  slice.SliceType.Normal = sliceNormal

  # update the view to ensure updated data information
  #renderView.Update()

  # set scalar coloring
  ColorBy(sliceDisplay,('POINTS',var))

  # rescale color and/or opacity maps used to include current data range
  sliceDisplay.RescaleTransferFunctionToDataRange(True,False)

  # show color bar/color legend
  sliceDisplay.SetScalarBarVisibility(renderView,True)

  # Color 
  colorTF = GetColorTransferFunction(var)
  opacityTF = GetOpacityTransferFunction(var)

  # Rescale color bar
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

  # Colorbar
  colorbar = GetScalarBar(colorTF,renderView)
  colorbar.WindowLocation = 'UpperCenter'
  colorbar.Orientation = 'Horizontal'
  colorbar.TitleJustification = 'Centered'
  colorbar.ScalarBarLength = 0.33
  colorbar.TitleBold = 1
  colorbar.TitleItalic = 0
  colorbar.TitleShadow = 1
  colorbar.TitleFontSize = 20
  colorbar.LabelBold = 1
  colorbar.LabelItalic = 0
  colorbar.LabelShadow = 1
  colorbar.LabelFontSize = 15

  # Get rid of default color bar -- must come after main var
  if (var != 'rho'):
    GetColorTransferFunction('rho').RescaleTransferFunction(0.0,1.0)
    GetOpacityTransferFunction('rho').RescaleTransferFunction(0.0,1.0)
    HideScalarBarIfNotNeeded(GetColorTransferFunction('rho'),renderView)

  # Camera, view, lighting
  renderView.CameraPosition = [camera[0],camera[1],camera[2]]
  renderView.CameraFocalPoint = [camera[0],camera[1],0.0]
  renderView.CameraViewAngle = 1.0
  GetRenderView().UseLight = 0
  GetRenderView().ViewSize = [pixels[0],pixels[1]]

  # Hide mesh outline and axis
  dataDisplay.Opacity = 0.0
  renderView.OrientationAxesVisibility = 0

  drawTime = time.time() - t0
  #==============================================================================================================
  # Render and write to file
  t0 = time.time()

  renderView.Update()
  Render()
  WriteImage(imageFile)

  writeTime = time.time() - t0
  print('--> File written: {}.  Read = {:4.1f}s, draw = {:4.1f}s, write = {:4.1f}s'
    .format(imageFile,readTime,drawTime,writeTime))

  #==============================================================================================================
  # Delete data to prevent accumulation between each outer loop iteration
  Delete(data)
  Delete(renderView)
  Delete(dataDisplay)
  Delete(materialLibrary)
  Delete(slice)
  Delete(colorTF)
  Delete(opacityTF)
  Delete(colorbar)
