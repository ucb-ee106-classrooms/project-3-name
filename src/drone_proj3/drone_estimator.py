import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 14


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][1] is the thrust of the quadrotor
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is translational position in x (m),
            x[i][1] is translational position in z (m),
            x[i][2] is the bearing (rad) of the quadrotor
            x[i][3] is translational velocity in x (m/s),
            x[i][4] is translational velocity in z (m/s),
            x[i][5] is angular velocity (rad/s),
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][1] is distance to the landmark (m)
            y[i][2] is relative bearing (rad) w.r.t. the landmark
        x_hat : list
            A list of estimated system states. It should follow the same format
            as x.
        dt : float
            Update frequency of the estimator.
        fig : Figure
            matplotlib Figure for real-time plotting.
        axd : dict
            A dictionary of matplotlib Axis for real-time plotting.
        ln* : Line
            matplotlib Line object for ground truth states.
        ln_*_hat : Line
            matplotlib Line object for estimated states.
        canvas_title : str
            Title of the real-time plot, which is chosen to be estimator type.

    Notes
    ----------
        The landmark is positioned at (0, 5, 5).
    """
    # noinspection PyTypeChecker
    def __init__(self, is_noisy=False):
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.t = []
        self.fig, self.axd = plt.subplot_mosaic(
            [['xz', 'phi'],
             ['xz', 'x'],
             ['xz', 'z']], figsize=(20.0, 10.0))
        self.ln_xz, = self.axd['xz'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xz_hat, = self.axd['xz'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_z, = self.axd['z'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_z_hat, = self.axd['z'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'

        # Defined in dynamics.py for the dynamics model
        # m is the mass and J is the moment of inertia of the quadrotor 
        self.gr = 9.81 
        self.m = 0.92
        self.J = 0.0023
        # These are the X, Y, Z coordinates of the landmark
        self.landmark = (0, 5, 5)

        # This is a (N,12) where it's time, x, u, then y_obs 
        if is_noisy:
            with open('noisy_data.npy', 'rb') as f:
                self.data = np.load(f)
        else:
            with open('data.npy', 'rb') as f:
                self.data = np.load(f)

        self.dt = self.data[-1][0]/self.data.shape[0]


    def run(self):
        for i, data in enumerate(self.data):
            self.t.append(np.array(data[0]))
            self.x.append(np.array(data[1:7]))
            self.u.append(np.array(data[7:9]))
            self.y.append(np.array(data[9:12]))
            if i == 0:
                self.x_hat.append(self.x[-1])
            else:
                self.update(i)
        return self.x_hat

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xz'].set_title(self.canvas_title)
        self.axd['xz'].set_xlabel('x (m)')
        self.axd['xz'].set_ylabel('z (m)')
        self.axd['xz'].set_aspect('equal', adjustable='box')
        self.axd['xz'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].set_xlabel('t (s)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].set_xlabel('t (s)')
        self.axd['x'].legend()
        self.axd['z'].set_ylabel('z (m)')
        self.axd['z'].set_xlabel('t (s)')
        self.axd['z'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xzline(self.ln_xz, self.x)
        self.plot_xzline(self.ln_xz_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_zline(self.ln_z, self.x)
        self.plot_zline(self.ln_z_hat, self.x_hat)

    def plot_xzline(self, ln, data):
        if len(data):
            x = [d[0] for d in data]
            z = [d[1] for d in data]
            ln.set_data(x, z)
            self.resize_lim(self.axd['xz'], x, z)

    def plot_philine(self, ln, data):
        if len(data):
            t = self.t
            phi = [d[2] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = self.t
            x = [d[0] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_zline(self, ln, data):
        if len(data):
            t = self.t
            z = [d[1] for d in data]
            ln.set_data(t, z)
            self.resize_lim(self.axd['z'], t, z)

    # noinspection PyMethodMayBeStatic
    def resize_lim(self, ax, x, y):
        xlim = ax.get_xlim()
        ax.set_xlim([min(min(x) * 1.05, xlim[0]), max(max(x) * 1.05, xlim[1])])
        ylim = ax.get_ylim()
        ax.set_ylim([min(min(y) * 1.05, ylim[0]), max(max(y) * 1.05, ylim[1])])

class OracleObserver(Estimator):
    """Oracle observer which has access to the true state.

    This class is intended as a bare minimum example for you to understand how
    to work with the code.

    Example
    ----------
    To run the oracle observer:
        $ python drone_estimator_node.py --estimator oracle_observer
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Oracle Observer'

    def update(self, _):
        self.x_hat.append(self.x[-1])


class DeadReckoning(Estimator):
    """Dead reckoning estimator.

    Your task is to implement the update method of this class using only the
    u attribute and x0. You will need to build a model of the unicycle model
    with the parameters provided to you in the lab doc. After building the
    model, use the provided inputs to estimate system state over time.

    The method should closely predict the state evolution if the system is
    free of noise. You may use this knowledge to verify your implementation.

    Example
    ----------
    To run dead reckoning:
        $ python drone_estimator_node.py --estimator dead_reckoning
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Dead Reckoning'

    def update(self, _):
        if len(self.x_hat) > 0:
            last_state = self.x_hat[-1]
            
            # Last input
            u = self.u[_]
            
            x = last_state[0]
            z = last_state[1]
            phi = last_state[2]
            vx = last_state[3]
            vz = last_state[4]
            omega = last_state[5]
            
            u1 = u[0]
            u2 = u[1]
            
            # x[t+1] = x[t] + f(x[t], u[t]) * dt
            new_x = x + vx * self.dt
            new_z = z + vz * self.dt
            new_phi = phi + omega * self.dt
            new_vx = vx + (-u1 * np.sin(phi) / self.m) * self.dt
            new_vz = vz + (-self.gr + u1 * np.cos(phi) / self.m) * self.dt
            new_omega = omega + (u2 / self.J) * self.dt
            
            # New state estimate
            new_state = np.array([new_x, new_z, new_phi, new_vx, new_vz, new_omega])
            self.x_hat.append(new_state)

# noinspection PyPep8Naming
class ExtendedKalmanFilter(Estimator):
    """Extended Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    unicycle model and linearize it at every operating point. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive extended Kalman filter update rule.

    Hint: You may want to reuse your code from DeadReckoning class and
    KalmanFilter class.

    Attributes:
    ----------
        landmark : tuple
            A tuple of the coordinates of the landmark.
            landmark[0] is the x coordinate.
            landmark[1] is the y coordinate.
            landmark[2] is the z coordinate.

    Example
    ----------
    To run the extended Kalman filter:
        $ python drone_estimator_node.py --estimator extended_kalman_filter
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Extended Kalman Filter'

        self.A = None
        self.B = None
        self.C = None

        self.Q = np.array([
            [0.01, 0, 0, 0, 0, 0],
            [0, 0.01, 0, 0, 0, 0],
            [0, 0, 0.01, 0, 0, 0],
            [0, 0, 0, 0.1, 0, 0],
            [0, 0, 0, 0, 0.1, 0],
            [0, 0, 0, 0, 0, 0.1]
            ])

        self.R = np.array([
            [0.1, 0],
            [0, 0.001]
            ])
        
        self.P = np.array([
            [0.1, 0, 0, 0, 0, 0],
            [0, 0.1, 0, 0, 0, 0],
            [0, 0, 0.1, 0, 0, 0],
            [0, 0, 0, 0.1, 0, 0],
            [0, 0, 0, 0, 0.1, 0],
            [0, 0, 0, 0, 0, 0.1]
            ])

    # noinspection DuplicatedCode
    def update(self, i):
        if len(self.x_hat) > 0: #and self.x_hat[-1][0] < self.x[-1][0]:

            # State extrapolation
            last_state = self.x_hat[-1]
            u = self.u[i]
            x_pred = self.g(last_state, u)
            
            # Dynamics linearization - compute Jacobian A
            A = self.A(last_state, u)
            
            # Covariance extrapolation
            P_pred = A @ self.P @ A.T + self.Q
            
            # Measurement linearization - compute Jacobian C
            C = self.C(x_pred)
            
            # Kalman gain
            K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + self.R)
            
            # State update
            x_updated = x_pred + K @ (self.y[i] - self.h(x_pred))
            
            # Covariance update
            I = np.eye(len(self.P))
            self.P = (I - K @ C) @ P_pred
            
            # New state estimate
            self.x_hat.append(x_updated)

    def g(self, x, u):
        """
        Dynamics model function for the planar quadrotor.
        
        Parameters:
        x : current state [x, z, phi, vx, vz, omega]
        u : control input [u1 (thrust), u2 (moment)]
        
        Returns:
        The predicted next state using forward Euler integration
        """
        A = np.array([x[3], x[4], x[5], 0, -self.gr, 0])

        phi = x[2]
        B = np.array([[0, 0],
                    [0, 0],
                    [0, 0],
                    [-np.sin(phi)/self.m, 0],
                    [np.cos(phi)/self.m,  0],
                    [0, 1/self.J]])
        
        x_dot = A + B @ u 
        
        x_new = x + x_dot * self.dt
        
        return x_new


    def h(self, x):
        """
        Measurement model function.
        
        Parameters:
        x : state [x, z, phi, vx, vz, omega]
        
        Returns:
        The predicted measurement [distance to landmark, bearing]
        """
        # drone state
        drone_x = x[0]
        drone_z = x[1]
        drone_phi = x[2]
        
        # landmark position
        landmark_x = self.landmark[0]
        landmark_z = self.landmark[2]
        
        distance = np.sqrt((landmark_x - drone_x)**2 + (landmark_z - drone_z)**2)
        
        return np.array([distance, drone_phi])

    def A(self, x, u):
        """
        Approximate the Jacobian matrix A of the dynamics model at the current state.
        
        Parameters:
        x : current state
        u : control input
        
        Returns:
        The linearized dynamics matrix A
        """
        phi = x[2]
        u1 = u[0] # thrust force
        
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, -u1 * np.cos(phi) / self.m, 0, 0, 0],
            [0, 0, -u1 * np.sin(phi) / self.m, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        return A * self.dt

    
    def C(self, x):
        """
        Approximate the Jacobian matrix C of the measurement model at the current state.
        
        Parameters:
        x : current state
        
        Returns:
        The linearized measurement matrix C
        """
        # drone state
        drone_x = x[0]
        drone_z = x[1]
        
        # landmark position
        landmark_x = self.landmark[0]
        landmark_z = self.landmark[2]
        
        dx = landmark_x - drone_x
        dz = landmark_z - drone_z
        distance = np.sqrt(dx**2 + dz**2)
        
        C = np.array([[-dx / distance, -dz / distance, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])
        
        return C * self.dt
