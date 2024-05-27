from paraview.simple import (
    _DisableFirstRenderCameraReset,
    XMLPartitionedUnstructuredGridReader,
    GetActiveViewOrCreate,
    GetDisplayProperties,
    GetMaterialLibrary,
    GetSettingsProxy,
    Calculator,
    Gradient,
    Delete,
    Show,
    Transform,
    ColorBy,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    GetRenderView,
    Render,
    WriteImage,
    Slice
)
import time
import numpy as np


class SliceData():
    def __init__(self, dataName, dataRange, camera, colorScheme,
                 logScale, invert, pixels, normal, origin,
                 cbTitle, cbBackground=0, cbLabels=None, cbLabelFormat=None,
                 blueRedDivergentColorCenter=None,
                 backgroundColor=[0., 0., 0.],
                 rotation=None, reflection=None,
                 computeGradient=0,
                 vectorComponent=None,
                 prefix=""):

        self.dataName = dataName
        self.dataRange = dataRange
        self.camera = camera
        self.colorScheme = colorScheme
        self.logScale = logScale
        self.invert = invert
        self.cbTitle = cbTitle
        self.cbLabels = cbLabels
        self.cbLabelFormat = cbLabelFormat
        self.cbBackground = cbBackground
        self.backgroundColor = backgroundColor
        self.blueRedDivergentColorCenter = blueRedDivergentColorCenter
        self.pixels = pixels
        self.normal = normal
        self.origin = origin
        self.rotation = rotation
        self.reflection = reflection
        self.computeGradient = computeGradient
        self.vectorComponent = vectorComponent

        self.hasPrefix = False
        if prefix:
            self.prefix = prefix
            self.hasPrefix = True

    def __repr__(self):
        return "%s(dataName=%r, dataRange=%r, camera=%r, origin=%r, normal=%r, colorScheme=%r)" % (
            self.__class__.__name__, self.dataName, self.dataRange,
            self.camera, self.origin, self.normal, self.colorScheme)


def SimpleSlice3D(dir, iter, solutionData, sliceData):

    varName = sliceData.dataName
    camera_position = sliceData.camera
    varRange = sliceData.dataRange
    pixels = sliceData.pixels
    sliceNormal = sliceData.normal
    sliceOrigin = sliceData.origin

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
    solutionFile = solutionData

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

    # create a new Slice
    slice = Slice(Input=var)
    slice.SliceType.Origin = sliceData.origin
    slice.SliceType.Normal = sliceData.normal

    #origin_x = [0.63, 0, 0]
    #normal_x = [1.0, 0, 0]
    #slice_x = Slice(Input=var)
    #slice_x.SliceType.Origin = origin_x
    #slice_x.SliceType.Normal = normal_x

    sliceDisplay = Show(slice, renderView)
    ColorBy(sliceDisplay, ("POINTS", varName))
    sliceDisplay.SetScalarBarVisibility(renderView, True)

    #sliceDisplay_x = Show(slice_x, renderView)
    #ColorBy(sliceDisplay_x, ("POINTS", varName))
    #sliceDisplay_x.SetScalarBarVisibility(renderView, False)
#
    # Gets rid of "Failed to determine the LookupTable being used" error
    #calculatorDisplay = Show(var, renderView)
    #ColorBy(calculatorDisplay, ("POINTS", varName))
    #calculatorDisplay.SetScalarBarVisibility(renderView, True)
    ##Hide(var,renderView)

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
    colorbar.TitleShadow = 0
    colorbar.TitleFontSize = 20
    colorbar.LabelBold = 1
    colorbar.LabelItalic = 0
    colorbar.LabelShadow = 1
    colorbar.LabelFontSize = 15
    colorbar.ScalarBarLength = 0.33
    colorbar.WindowLocation = "Upper Center"
    colorbar.Orientation = "Horizontal"
    colorbar.DrawScalarBarOutline = 1
    colorbar.ScalarBarOutlineColor = [1., 1., 1.]

    drawTime = time.time() - t0 - readTime

    # Adjust view, render, and write

    # Camera, view, lighting
    #print(f"camera {camera}")
    renderView.ResetCamera()
    renderView.InteractionMode = "3D"
    camera_position = [0.4848529242829971, 0.04074485744518312, 0.14925796932969887]
    renderView.CameraPosition = camera_position
    camera_focus = [0.6034246984894842, 0.0007759475823024825, 0.028860359869871927]
    renderView.CameraFocalPoint = camera_focus
    camera_view_up = [0.022331628356743072, 0.9552023813959811, -0.29510965580675785]
    renderView.CameraViewUp = camera_view_up
    #renderView.CameraViewAngle = 30.0
    renderView.CameraParallelScale = 0.05
    GetRenderView().UseLight = 1
    pixels = [1300, 700]
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


def SimpleSlice(dir, iter, solutionData, sliceData):

    varName = sliceData.dataName
    camera = sliceData.camera
    varRange = sliceData.dataRange
    pixels = sliceData.pixels
    sliceNormal = sliceData.normal
    sliceOrigin = sliceData.origin

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
    solutionFile = solutionData

    # If func not provided, simply set it to varName -- everything is a Calculator
    #if (not func):
    func = varName

    # disable automatic camera reset on 'Show'
    _DisableFirstRenderCameraReset()

    # Read solution file
    t0 = time.time()

    # Open XDMF
    data = XMLPartitionedUnstructuredGridReader(FileName=[solutionFile])
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

    # Compute the gradient of the field in question
    # We rename the gradient data to the original variable name
    # all the processing that follows will use the new data
    grad_data = var
    if sliceData.computeGradient > 0:
        grad_data = Gradient(registrationName="varGrad", Input=data)
        grad_data.ScalarArray = ["POINTS", varName]
        grad_data.ResultArrayName = varName

    filter_data = grad_data
    if sliceData.rotation is not None:
        filter_data = Transform(Input=filter_data)
        filter_data.Transform.Rotate = sliceData.rotation

    display = Show(filter_data, renderView)

    if sliceData.reflection is not None:
        reflection_filter_data = Transform(Input=filter_data)
        reflection_filter_data.Transform.Scale = sliceData.reflection
        display_reflection = Show(reflection_filter_data, renderView)
        if sliceData.vectorComponent is not None:
            ColorBy(display_reflection, ("POINTS", varName, sliceData.vectorComponent))
        else:
            ColorBy(display_reflection, ("POINTS", varName))

    if sliceData.vectorComponent is not None:
        ColorBy(display, ("POINTS", varName, sliceData.vectorComponent))
    else:
        ColorBy(display, ("POINTS", varName))

    display.SetScalarBarVisibility(renderView, True)

    # set the background color
    if sliceData.backgroundColor is not None:
        colorPalette = GetSettingsProxy('ColorPalette')
        colorPalette.Background = sliceData.backgroundColor

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

    # custom divergent color mapping
    # centers the data at a particular value, along with the defined min and max
    if sliceData.blueRedDivergentColorCenter is not None:
        colorTF.RGBPoints = \
            [varRange[0], 0.23137254902, 0.298039215686, 0.752941176471,
             sliceData.blueRedDivergentColorCenter, 0.865, 0.865, 0.865,
             varRange[1], 0.705882352941, 0.0156862745098, 0.149019607843]
        colorTF.ColorSpace = "Diverging"

    # Colorbar
    colorbar = GetScalarBar(colorTF, renderView)
    colorbar.Title = sliceData.cbTitle
    colorbar.TitleJustification = "Centered"
    colorbar.DrawBackground = sliceData.cbBackground
    colorbar.TitleBold = 1

    if sliceData.cbLabelFormat is not None:
        colorbar.AutomaticLabelFormat = 0
        colorbar.LabelFormat = sliceData.cbLabelFormat
        colorbar.RangeLabelFormat = sliceData.cbLabelFormat
        #colorbar.RangeLabelFormat = '%-#6.1f'

    if sliceData.cbLabels is not None:
        colorbar.UseCustomLabels = 1
        colorbar.CustomLabels = sliceData.cbLabels


    colorbar.TitleItalic = 0
    colorbar.TitleShadow = 0
    colorbar.TitleFontSize = 24
    colorbar.LabelBold = 1
    colorbar.LabelItalic = 0
    colorbar.LabelShadow = 0
    colorbar.LabelFontSize = 20
    colorbar.ScalarBarLength = 0.33
    colorbar.WindowLocation = "Upper Center"
    colorbar.Orientation = "Horizontal"
    colorbar.DrawScalarBarOutline = 1
    colorbar.ScalarBarOutlineColor = [0., 0., 0.]

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
