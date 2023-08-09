from paraview.simple import (
    _DisableFirstRenderCameraReset,
    XMLPartitionedUnstructuredGridReader,
    GetActiveViewOrCreate,
    GetDisplayProperties,
    GetMaterialLibrary,
    Calculator,
    Delete,
    Show,
    ColorBy,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    GetRenderView,
    Render,
    WriteImage
)
import time
import numpy as np


class SliceData():
    def __init__(self, dataName, dataRange, camera, colorScheme,
                 logScale, invert, cbTitle, pixels, normal, origin,
                 prefix=""):
        self.dataName = dataName
        self.dataRange = dataRange
        self.camera = camera
        self.colorScheme = colorScheme
        self.logScale = logScale
        self.invert = invert
        self.cbTitle = cbTitle
        self.pixels = pixels
        self.normal = normal
        self.origin = origin

        self.hasPrefix = False
        if prefix:
            self.prefix = prefix
            self.hasPrefix = True

    def __repr__(self):
        return "%s(dataName=%r, dataRange=%r, camera=%r, colorScheme=%r)" % (
            self.__class__.__name__, self.dataName, self.dataRange,
            self.camera, self.colorScheme)


#def SimpleSlice(dir, iter, varName="cv_mass", varRange=None, camera=None,
                #colorScheme="Cool to Warm", logScale=0, invert=0,
                #func="", cbTitle="", prefix="",
                #pixels=None,
                #sliceNormal=None,
                #sliceOrigin=None):
def SimpleSlice(dir, iter, sliceData):

    varName = sliceData.dataName
    camera = sliceData.camera

    #if camera is None:
        #camera = [0.625, -0.0, 0.023]

    varRange = sliceData.dataRange

    #if varRange is None:
        #camera = [0.02, 0.1]

    pixels = sliceData.pixels

    #if pixels is None:
        #pixels = [1200, 600]
        ##pixels = np.zeros(shape=(2,))
        ##pixels[0] = 1200
        ##pixels[1] = 600

    sliceNormal = sliceData.normal
    #if sliceNormal is None:
        #sliceNormal = [0., 0., 1.0]
        ##sliceNormal[2] = 1.0

    sliceOrigin = sliceData.origin
    #if sliceOrigin is None:
        ##sliceOrigin = np.zeros(shape=(3,))
        #sliceOrigin = [0., 0., 0.]

    import os
    slice_img_dir = dir + "/slice_img"

    if not os.path.exists(slice_img_dir):
        os.makedirs(slice_img_dir)

    # Include prefix in output file name, if provided
    if sliceData.hasPrefix:
        prefix = sliceData.prefix
        imageFile = f"{slice_img_dir}/{prefix}-{varName}_{iter:09d}.png"
    else:
        imageFile = f"{slice_img_dir}/{varName}_{iter:09d}.png"
    solutionFile = "{}/prediction-fluid-{:09d}.pvtu".format(dir, iter)
    #solutionFileWall = "{}/prediction-wall-{:09d}.pvtu".format(dir, iter)

    # If func not provided, simply set it to varName -- everything is a Calculator
    #if (not func):
    func = varName

    # disable automatic camera reset on 'Show'
    _DisableFirstRenderCameraReset()

    #print('func={},prefix={},camera={},pixels={}'.format(func,prefix,camera,pixels))

    # Read solution file
    t0 = time.time()

    # Open XDMF
    data = XMLPartitionedUnstructuredGridReader(FileName=[solutionFile])
    #data.PointArrayStatus = ["cv_mass", "cv_energy", "cv_momentum",
                             #"dv_temperature", "dv_pressure",
                             #"mach", "velocity", "sponge_sigma",
                             #"cfl",
                             #"Y_C2H4", "Y_H2", "Y_H2O",
                             #"Y_O2", "Y_CO", "Y_fuel", "Y_air", "mu"]
    data.PointArrayStatus = varName
    readTime = time.time() - t0

    # Properties modified on data

    # get active view
    renderView = GetActiveViewOrCreate("RenderView")

    # display properties
    mainDisplay = GetDisplayProperties(data)
    mainDisplay.Representation = "Outline"
    #materialLibrary = GetMaterialLibrary()
    mainDisplay.SetScalarBarVisibility(renderView, True)

    # Calculator filter
    var = Calculator(Input=data)
    var.ResultArrayName = varName
    var.Function = func
    Delete(data)

    # Gets rid of "Failed to determine the LookupTable being used" error
    calculatorDisplay = Show(var, renderView)
    ColorBy(calculatorDisplay, ("POINTS", varName))
    calculatorDisplay.SetScalarBarVisibility(renderView, True)
    #Hide(var,renderView)

    #dataObject = Show(data, renderView, 'UnstructuredGridRepresentation')
    #dataObject.Representation = 'Surface'

    #ColorBy(mainDisplay, ('POINTS', 'mach'))

    # Color scheme settings
    colorTF = GetColorTransferFunction(varName)
    opacityTF = GetOpacityTransferFunction(varName)
    colorTF.ApplyPreset(sliceData.colorScheme, True)
    colorTF.RescaleTransferFunction(varRange[0], varRange[1])
    opacityTF.RescaleTransferFunction(varRange[0], varRange[1])

    # Log scale -- comes after rescaling color bar
    if (sliceData.logScale == 1):
        colorTF.MapControlPointsToLogSpace()
        colorTF.UseLogScale = 1

    # Invert color scheme, if requested
    invert = sliceData.invert
    if (invert == 1):
        colorTF.InvertTransferFunction()

    # Colorbar
    colorbar = GetScalarBar(colorTF, renderView)
    colorbar.Title = sliceData.cbTitle
    colorbar.TitleJustification = "Centered"
    colorbar.TitleBold = 1
    colorbar.TitleItalic = 0
    colorbar.TitleShadow = 1
    colorbar.TitleFontSize = 20
    colorbar.LabelBold = 1
    colorbar.LabelItalic = 0
    colorbar.LabelShadow = 1
    colorbar.LabelFontSize = 15
    colorbar.ScalarBarLength = 0.33
    colorbar.WindowLocation = "Upper Center"
    colorbar.Orientation = "Horizontal"

    drawTime = time.time() - t0 - readTime

    # Adjust view, render, and write

    # Camera, view, lighting
    #print(f"camera {camera}")
    renderView.ResetCamera()
    renderView.InteractionMode = "2D"
    renderView.CameraPosition = [camera[0], camera[1], camera[2]]
    renderView.CameraFocalPoint = [camera[0], camera[1], 0.0]
    renderView.CameraViewAngle = 1.0
    renderView.CameraParallelScale = camera[2]
    GetRenderView().UseLight = 0
    GetRenderView().ViewSize = [pixels[0], pixels[1]]

    # Hide mesh outline and axis
    #mainDisplay.Opacity = 0.0
    renderView.OrientationAxesVisibility = 0

    # Render and write to file
    renderView.Update()
    Render()
    WriteImage(imageFile)

    writeTime = time.time() - t0 - drawTime - readTime
    print(f"--> File written: {imageFile}.",
          f" Read = {readTime:4.1f}s,",
          f" Draw = {drawTime:4.1f}s,",
          f" Write = {writeTime:4.1f}s"
          )

    # Delete data to prevent accumulation between each outer loop iteration
    Delete(var)
    Delete(renderView)
    Delete(mainDisplay)
    #Delete(materialLibrary)
    Delete(colorTF)
    Delete(opacityTF)
    Delete(colorbar)
