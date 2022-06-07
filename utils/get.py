import utils.mapping as mapping
# import feature_scaling
import openfoamparser as ofpp
import matplotlib.pyplot as plt
import numpy as np
import math
import vtk


def interior_data(addr, turbulence):
    u = np.float32(ofpp.parse_internal_field(addr + "/U"))
    p = np.float32(ofpp.parse_internal_field(addr + "/p"))

    nu_tilda = 0
    if turbulence:
        nu_tilda = np.float32(ofpp.parse_internal_field(addr + "/nu"))

    ux = u[:, 0]
    uy = u[:, 2]

    return ux, uy, p, nu_tilda


def boundary_data(addr, turbulence):
    zone = vtk.PolyData(addr)
    u = zone.cell_arrays['U']
    p = zone.cell_arrays['p']

    nu_tilda = 0
    if turbulence:
        nu_tilda = zone.cell_arrays['nu']

    ux = u[:, 0]
    uy = u[:, 2]

    return ux, uy, p, nu_tilda


def single_sample(grid, interior_addr, bottom_addr, top_addr, dim, turbulence, pos):
    height = int(dim[0])
    length = dim[2]
    viscosity = dim[3]

    ux_interior, uy_interior, p_interior, nu_tilda_interior = interior_data(interior_addr, turbulence)

    #  Ux_bottom,   Uy_bottom,   p_bottom, nuTilda_bottom     = boundaryData(bottom_addr, turb)
    #  Ux_top,      Uy_top,      p_top, nuTilda_top           = boundaryData(top_addr, turb)

    # Ux_bottom,   Uy_bottom,   p_bottom, nuTilda_bottom     = boundaryData(interior_addr, turb)
    # Ux_top,      Uy_top,      p_top, nuTilda_top           = boundaryData(interior_addr, turb)

    ux_interior = mapping.interior(ux_interior, dim, grid)
    uy_interior = mapping.interior(uy_interior, dim, grid)
    p_interior = mapping.interior(p_interior, dim, grid)

    ux = ux_interior
    uy = uy_interior
    p = p_interior

    #  if pos == "input":

    #   boundary="bottom"
    #   Ux_bottom = mapping.boundary(Ux_bottom, dim, grid, ux_interior, boundary)
    #   Uy_bottom = mapping.boundary(Uy_bottom, dim, grid, uy_interior, boundary)
    #   p_bottom  = mapping.boundary(p_bottom,  dim, grid, p_interior, boundary)

    #   boundary="top"
    #   Ux_top = mapping.boundary(Ux_top, dim, grid, ux_interior, boundary)
    #   Uy_top = mapping.boundary(Uy_top, dim, grid, uy_interior, boundary)
    #   p_top  = mapping.boundary(p_top,  dim, grid, p_interior, boundary)

    if turbulence:
        nu_tilda_interior = mapping.interior(nu_tilda_interior, dim, grid)
        nu_tilda = nu_tilda_interior

    #   if pos == "input":
    #    boundary="bottom"
    # #   nuTilda_bottom = mapping.boundary(nuTilda_bottom, dim, grid,nu_tilda_interior,boundary)
    #    boundary="top"
    #  #  nuTilda_top = mapping.boundary(nuTilda_top, dim, grid, nu_tilda_interior,boundary)
    #
    #  if pos == "input":
    #
    # #  Ux = np.append(Ux_bottom, ux_interior, axis = 0)
    # #  Uy = np.append(Uy_bottom, uy_interior, axis = 0)
    # #  p  = np.append(p_bottom, p_interior,   axis = 0)
    #
    #   if (turb):
    #  #  nuTilda = np.append(nuTilda_bottom, nu_tilda_interior, axis = 0)
    #
    #  # Ux = np.append(Ux, Ux_top,  axis = 0)
    # #  Uy = np.append(Uy, Uy_top,  axis = 0)
    #  # p  = np.append(p,  p_top,   axis = 0)
    #
    #   if (turb):
    #   # nuTilda = np.append(nuTilda, nuTilda_top,  axis = 0)

    ux = ux.reshape([ux.shape[0], ux.shape[1], 1])
    uy = uy.reshape([uy.shape[0], uy.shape[1], 1])
    p = p.reshape([p.shape[0], p.shape[1], 1])

    if turbulence:
        nu_tilda = nu_tilda.reshape([nu_tilda.shape[0], nu_tilda.shape[1], 1])

    if grid == "1b_rect_grid":
        ux_avg = ux[int(height / 2), 0, 0]
        uy_avg = uy[int(height / 2), 0, 0]
        u_avg = ux_avg
        nu_tilda_avg = dim[3]

    elif grid == "ellipse":
        # main_ux = ux[height+1,0,0]
        # main_uy = uy[height+1,0,0]
        u_avg = 0.6
        nu_tilda_avg = dim[3]

    elif grid == "airfoil":
        u_avg = ux[height + 1, 0, 0]
        nu_tilda_avg = dim[3]

    if pos == "input" or pos == "output":

        ux /= u_avg
        uy /= u_avg
        p /= u_avg * u_avg

        if turbulence:
            nu_tilda /= nu_tilda_avg

    data = np.concatenate((ux, uy), axis=2)
    data = np.concatenate((data, p), axis=2)

    if turbulence:
        data = np.concatenate((data, nu_tilda), axis=2)

    return data


def case_data(x_addrs, y_addr, coordinates, dim, grid, turb, x_train, y_train):
    x_addrs = x_addrs[0]
    n = len(x_addrs)

    y_interior_addr = y_addr[0]
    y_interior_addr = y_interior_addr[0]
    y_top_addr = y_interior_addr
    y_bottom_addr = y_interior_addr

    for i in range(0, n):
        x_interior_addr = x_addrs[i]
        x_bottom_addr = []
        x_top_addr = []

        pos = "input"
        data_cell = single_sample(grid, x_interior_addr,
                                  x_bottom_addr, x_top_addr,
                                  dim, turb, pos)

        x_train.append(data_cell)

        pos = "output"
        data_cell = single_sample(grid, y_interior_addr,
                                  y_bottom_addr, y_top_addr,
                                  dim, turb, pos)

        y_train.append(data_cell)

    return
