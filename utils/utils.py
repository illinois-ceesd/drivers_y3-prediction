"""mirgecom driver initializer for the Y2 prediction."""
import numpy as np


def getIsentropicPressure(mach, P0, gamma):
    pressure = (1. + (gamma - 1.)*0.5*mach**2)
    pressure = P0*pressure**(-gamma / (gamma - 1.))
    return pressure


def getIsentropicTemperature(mach, T0, gamma):
    temperature = (1. + (gamma - 1.)*0.5*mach**2)
    temperature = T0/temperature
    return temperature


def getMachFromAreaRatio(area_ratio, gamma, mach_guess=0.01):
    error = 1.0e-8
    nextError = 1.0e8
    g = gamma
    M0 = mach_guess
    while nextError > error:
        R = (((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))/M0
            - area_ratio)
        dRdM = (2*((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               / (2*g - 2)*(g - 1)/(2/(g + 1) + ((g - 1)/(g + 1)*M0*M0)) -
               ((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               * M0**(-2))
        M1 = M0 - R/dRdM
        nextError = abs(R)
        M0 = M1

    return M1


def get_y_from_x(x, data):
    """
    Return the linearly interpolated the value of y
    from the value in data(x,y) at x
    """

    if x <= data[0][0]:
        y = data[0][1]
    elif x >= data[-1][0]:
        y = data[-1][1]
    else:
        ileft = 0
        iright = data.shape[0]-1

        # find the bracketing points, simple subdivision search
        while iright - ileft > 1:
            ind = int(ileft+(iright - ileft)/2)
            if x < data[ind][0]:
                iright = ind
            else:
                ileft = ind

        leftx = data[ileft][0]
        rightx = data[iright][0]
        lefty = data[ileft][1]
        righty = data[iright][1]

        dx = rightx - leftx
        dy = righty - lefty
        y = lefty + (x - leftx)*dy/dx
    return y


def get_theta_from_data(data):
    """
    Calculate theta = arctan(dy/dx)
    Where data[][0] = x and data[][1] = y
    """

    theta = data.copy()
    for index in range(1, theta.shape[0]-1):
        #print(f"index {index}")
        theta[index][1] = np.arctan((data[index+1][1]-data[index-1][1]) /
                          (data[index+1][0]-data[index-1][0]))
    theta[0][1] = np.arctan(data[1][1]-data[0][1])/(data[1][0]-data[0][0])
    theta[-1][1] = np.arctan(data[-1][1]-data[-2][1])/(data[-1][0]-data[-2][0])
    return (theta)


def smooth_step(actx, x, epsilon=1e-12):
    # return actx.np.tanh(x)
    # return actx.np.where(
    #     actx.np.greater(x, 0),
    #     actx.np.tanh(x)**3,
    #     0*x)
    return (
        actx.np.greater(x, 0) * actx.np.less(x, 1) * (1 - actx.np.cos(np.pi*x))/2
        + actx.np.greater(x, 1))
