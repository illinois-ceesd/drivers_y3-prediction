# configuration file for visualization

cavity_camera = [0.625, 0.0, 0.023]

from slice import SliceData
slice_data = []
slice_data_3d = []
slice_data.append(
    SliceData(
        dataName="cv_mass",
        dataRange=[0.02, 0.1],
        camera=cavity_camera,
        colorScheme="erdc_rainbow_dark",
        logScale=0,
        invert=0,
        cbTitle="Density [kg/m^3]",
        pixels=[1300, 700],
        normal=[0, 0, 1],
        origin=[0, 0, 0],
        prefix="cavity")
)

slice_data.append(
    SliceData(
        dataName="dv_pressure",
        dataRange=[1500.0, 10000],
        camera=cavity_camera,
        colorScheme="GREEN-WHITE_LINEAR",
        logScale=0,
        invert=1,
        cbTitle="Pressure [Pa]",
        pixels=[1300, 700],
        normal=[0, 0, 1],
        origin=[0, 0, 0],
        prefix="cavity")
)

slice_data.append(
    SliceData(
        dataName="dv_temperature",
        dataRange=[300.0, 1000],
        camera=cavity_camera,
        colorScheme="Black-Body Radiation",
        logScale=0,
        invert=0,
        cbTitle="Temperature [K]",
        pixels=[1300, 700],
        normal=[0, 0, 1],
        origin=[0, 0, 0],
        prefix="cavity")
)

slice_data.append(
    SliceData(
        dataName="mach",
        dataRange=[0.0, 3.5],
        camera=cavity_camera,
        colorScheme="Rainbow Desaturated",
        logScale=0,
        invert=0,
        cbTitle="Mach Number",
        pixels=[1300, 700],
        normal=[0, 0, 1],
        origin=[0, 0, 0],
        prefix="cavity")
)
