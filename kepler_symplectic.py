"""

    Integrate equation of motion of a particle moving in
    central 1/r potential (Kepler problem)
    using two (symlectic and non-symplectic) algorithms.

    The hamiltonian of the Kepler problem (dimensionless units):

        H = 1/2*(p_x**2 + p_y**2) - 1/sqrt(q_x**2 + q_y**2)

    Equation of motion:

        p_x' =-dH/dq_x = q_x/(sqrt(q_x**2 + q_y**2))**3 = g(q_x, q_y),
        p_y' =-dH/dq_y = q_y/(sqrt(q_x**2 + q_y**2))**3 = g(q_y, q_x),
        q_x' = dH/dp_x = p_x = f(p_x),
        q_y' = dH/dp_y = p_y = f(p_y).

"""

import numpy as np


def fff(mom):
    """ velocity """
    return mom


def ggg(qqa, qqb):
    """ acceleration """
    return -qqa/((np.sqrt(qqa**2 + qqb**2))**3)


def hamiltonian2d(pq0, npts, stepi, algo):
    """
    Solve the separable two-dimensional Hamiltonian system

        p_x' =-dH/dq_x = g(q_x, q_y),
        p_y' =-dH/dq_y = g(q_y, q_x),
        q_x' = dH/dp_x = p_x = f(p_x),
        q_y' = dH/dp_y = p_y = f(p_y).

        (px0, py0, qx0, qy0) = pq0

        algo - algorithm: 'e' - 'regular' Euler, else symplectic Euler
    """

    tgrid = stepi*np.linspace(0, npts, npts+1)

    ppx = np.zeros(npts+1, dtype=np.float64)
    ppy = np.zeros(npts+1, dtype=np.float64)
    qqx = np.zeros(npts+1, dtype=np.float64)
    qqy = np.zeros(npts+1, dtype=np.float64)

    ppx[0] = pq0[0]
    ppy[0] = pq0[1]
    qqx[0] = pq0[2]
    qqy[0] = pq0[3]

    for k in xrange(npts):
        pxold = ppx[k]
        pyold = ppy[k]
        qxold = qqx[k]
        qyold = qqy[k]

        qxnew = qxold + stepi*fff(pxold)
        qynew = qyold + stepi*fff(pyold)
        if algo == 'e':
            # regular Euler
            pxnew = pxold + stepi*ggg(qxold, qyold)
            pynew = pyold + stepi*ggg(qyold, qxold)
        else:
            # symplectic Euler
            pxnew = pxold + stepi*ggg(qxnew, qynew)
            pynew = pyold + stepi*ggg(qynew, qxnew)

        ppx[k+1] = pxnew
        ppy[k+1] = pynew
        qqx[k+1] = qxnew
        qqy[k+1] = qynew

    return tgrid, ppx, ppy, qqx, qqy


from matplotlib.pyplot import figure, show


def show_orbits(times, qqx, qqy, energy, angul, rungel, size):
    """
       Plot the phase space trajectory and the changes in energy,
       angular momentum, and Runge-Lenz vector vs time
    """

    fig = figure()

    ax1 = fig.add_subplot(2, 2, 1, adjustable='box', aspect=1.)
    ax1.plot(qqx, qqy)
    ax1.set_xlim([-size, size])
    ax1.set_ylim([-size, size])
    ax1.set_xlabel('qx')
    ax1.set_ylabel('qy')
    ax1.set_title('Phase space trajectory')
    ax1.grid()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, energy-energy[0])
    ax2.set_title('Energy change')
    ax2.set_xlabel('t')
    ax2.grid()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(times, angul - angul[0])
    ax2.set_xlabel('t')
    ax3.set_title('Angular momentum change')
    ax3.grid()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(times, rungel - rungel[0])
    ax4.set_xlabel('t')
    ax4.set_title('Runge Lenz vector change')
    ax4.grid()

    return fig

if __name__ == '__main__':

    import time

    NST = 10000                       # Number of steps
    TFIN = 100.                       # Final time, try 100, 200
    ECC = 0.3                         # Eccentricity of the orbit
    STEP = TFIN/NST
    METHODS = ('e', 's')

    """
    The initial conditions correspond to an orbit of period 2*pi
    and energy -1/2.

    p0_x = 0., p0_y = np.sqrt((1. + e)/(1. - e)), q0_x = 1. - e, q0_y = 0.
    """
    ICOND = np.array((0., np.sqrt((1. + ECC)/(1. - ECC)), 1. - ECC, 0.))

    for method in METHODS:

        print "\nmethod: %s" % method
        start_time = time.time()

        TIMES, XMOMENTA, YMOMENTA, XCOORDS, YCOORDS = \
               hamiltonian2d(ICOND, NST, STEP, method)

        print "Elapsed time: %s seconds" % (time.time() - start_time)

        ENERG = 0.5*(XMOMENTA**2 + YMOMENTA**2) \
                - 1./np.sqrt(XCOORDS**2 + YCOORDS**2)  # energy
        ANGUL = XCOORDS*YMOMENTA - YCOORDS*XMOMENTA    # angular momentum

        # Runge-Lenz vector
        RUNGEX = YMOMENTA*ANGUL - XCOORDS/np.sqrt(XCOORDS**2 + YCOORDS**2)
        RUNGEY = -XMOMENTA*ANGUL - YCOORDS/np.sqrt(XCOORDS**2 + YCOORDS**2)

        graph = show_orbits(TIMES, XCOORDS, YCOORDS, ENERG, ANGUL, RUNGEX, 4.)
        graph.show()

    show(block=False)
    raw_input()    # to keep the windows open

    from matplotlib.pyplot import close
    close('all')
