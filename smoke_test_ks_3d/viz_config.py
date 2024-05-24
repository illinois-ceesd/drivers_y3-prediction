# configuration file for visualization

cavity_camera = [0.52, 0.024, 0.077]

from slice import SliceData
# 2d surfaces
slice_data = []

# 3d slices
slice_data_3d = []
slice_data_3d.append(
    SliceData(
        dataName="dv_temperature",
        dataRange=[300.0, 1000],
        camera=cavity_camera,
        #colorScheme="Black-Body Radiation",
        colorScheme="Cool to Warm",
        logScale=0,
        invert=0,
        cbTitle="Temperature [K]",
        pixels=[1300, 700],
        normal=[0, 0, 1],
        origin=[0, 0, 0],
        prefix="cavity")
)
