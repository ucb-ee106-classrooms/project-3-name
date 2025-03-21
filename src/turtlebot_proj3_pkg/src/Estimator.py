import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['FreeSans', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 14


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        d : float
            Half of the track width (m) of TurtleBot3 Burger.
        r : float
            Wheel radius (m) of the TurtleBot3 Burger.
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][0] is timestamp (s),
            u[i][1] is left wheel rotational speed (rad/s), and
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is timestamp (s),
            x[i][1] is bearing (rad),
            x[i][2] is translational position in x (m),
            x[i][3] is translational position in y (m),
            x[i][4] is left wheel rotational position (rad), and
            x[i][5] is right wheel rotational position (rad).
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][0] is timestamp (s),
            y[i][1] is translational position in x (m) when freeze_bearing:=true,
            y[i][1] is distance to the landmark (m) when freeze_bearing:=false,
            y[i][2] is translational position in y (m) when freeze_bearing:=true, and
            y[i][2] is relative bearing (rad) w.r.t. the landmark when
            freeze_bearing:=false.
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
        sub_u : rospy.Subscriber
            ROS subscriber for system inputs.
        sub_x : rospy.Subscriber
            ROS subscriber for system states.
        sub_y : rospy.Subscriber
            ROS subscriber for system outputs.
        tmr_update : rospy.Timer
            ROS Timer for periodically invoking the estimator's update method.

    Notes
    ----------
        The frozen bearing is pi/4 and the landmark is positioned at (0.5, 0.5).
    """
    # noinspection PyTypeChecker
    def __init__(self):
        self.total_processing_time = 0
        self.total_position_error = 0
        self.d = 0.08
        self.r = 0.033
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.dt = 0.1
        self.fig, self.axd = plt.subplot_mosaic(
            [['xy', 'phi'],
             ['xy', 'x'],
             ['xy', 'y'],
             ['xy', 'thl'],
             ['xy', 'thr']], figsize=(20.0, 10.0))
        self.ln_xy, = self.axd['xy'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xy_hat, = self.axd['xy'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_y, = self.axd['y'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_y_hat, = self.axd['y'].plot([], 'o-c', label='Estimated')
        self.ln_thl, = self.axd['thl'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thl_hat, = self.axd['thl'].plot([], 'o-c', label='Estimated')
        self.ln_thr, = self.axd['thr'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thr_hat, = self.axd['thr'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'
        self.sub_u = rospy.Subscriber('u', Float32MultiArray, self.callback_u)
        self.sub_x = rospy.Subscriber('x', Float32MultiArray, self.callback_x)
        self.sub_y = rospy.Subscriber('y', Float32MultiArray, self.callback_y)
        self.tmr_update = rospy.Timer(rospy.Duration(self.dt), self.update)

    def callback_u(self, msg):
        self.u.append(msg.data)

    def callback_x(self, msg):
        self.x.append(msg.data)
        if len(self.x_hat) == 0:
            self.x_hat.append(msg.data)

    def callback_y(self, msg):
        self.y.append(msg.data)

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xy'].set_title(self.canvas_title)
        self.axd['xy'].set_xlabel('x (m)')
        self.axd['xy'].set_ylabel('y (m)')
        self.axd['xy'].set_aspect('equal', adjustable='box')
        self.axd['xy'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].legend()
        self.axd['y'].set_ylabel('y (m)')
        self.axd['y'].legend()
        self.axd['thl'].set_ylabel('theta L (rad)')
        self.axd['thl'].legend()
        self.axd['thr'].set_ylabel('theta R (rad)')
        self.axd['thr'].set_xlabel('Time (s)')
        self.axd['thr'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xyline(self.ln_xy, self.x)
        self.plot_xyline(self.ln_xy_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_yline(self.ln_y, self.x)
        self.plot_yline(self.ln_y_hat, self.x_hat)
        self.plot_thlline(self.ln_thl, self.x)
        self.plot_thlline(self.ln_thl_hat, self.x_hat)
        self.plot_thrline(self.ln_thr, self.x)
        self.plot_thrline(self.ln_thr_hat, self.x_hat)

    def plot_xyline(self, ln, data):
        if len(data):
            x = [d[2] for d in data]
            y = [d[3] for d in data]
            ln.set_data(x, y)
            self.resize_lim(self.axd['xy'], x, y)

    def plot_philine(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            phi = [d[1] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            x = [d[2] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_yline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            y = [d[3] for d in data]
            ln.set_data(t, y)
            self.resize_lim(self.axd['y'], t, y)

    def plot_thlline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thl = [d[4] for d in data]
            ln.set_data(t, thl)
            self.resize_lim(self.axd['thl'], t, thl)

    def plot_thrline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thr = [d[5] for d in data]
            ln.set_data(t, thr)
            self.resize_lim(self.axd['thr'], t, thr)

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
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=oracle_observer \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
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
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=dead_reckoning \
            noise_injection:=true \
            freeze_bearing:=false
    For debugging, you can simulate a noise-free unicycle model by setting
    noise_injection:=false.
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Dead Reckoning'

    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
    

            start_time = rospy.Time.now()  # Get the current time
            
            # Get the latest state estimate and time
            last_state = self.x_hat[-1]
            current_time = self.x[-1][0]
            last_time = last_state[0]
            
            # Latest input
            latest_input = None
            for u_data in reversed(self.u):
                if u_data[0] <= current_time:
                    latest_input = u_data
                    break
            
            if latest_input is None:
                self.x_hat.append(last_state)
                return
            
            # Input
            u_L = latest_input[1]
            u_R = latest_input[2]
            
            # Last estimated state
            phi = last_state[1]
            x_pos = last_state[2]
            y_pos = last_state[3]
            theta_L = last_state[4]
            theta_R = last_state[5]
            dt = current_time - last_time
            
            # Unicycle model: x[t+1] = x[t] + f(x[t], u[t]) * dt
            new_phi = phi + (self.r / (2 * self.d)) * (u_R - u_L) * dt
            new_x = x_pos + (self.r / 2) * (u_L + u_R) * np.cos(phi) * dt
            new_y = y_pos + (self.r / 2) * (u_L + u_R) * np.sin(phi) * dt
            new_theta_L = theta_L + u_L * dt
            new_theta_R = theta_R + u_R * dt
            
            # New state estimate
            new_state = [current_time, new_phi, new_x, new_y, new_theta_L, new_theta_R]
            self.x_hat.append(new_state)

            #calculate how much time it takes to process one step
            total_time_per_step = (rospy.Time.now() - start_time).to_sec()
            self.total_processing_time += total_time_per_step

            #calculate the difference between true state and estimated states in meters (only using x and y)
            true_x = self.x[-1][2]
            true_y = self.x[-1][3]


            print(len(self.x_hat))
            position_error = np.sqrt((true_x - new_x) ** 2 + (true_y - new_y) ** 2)
            self.total_position_error += position_error
            print("distance: ", position_error)
            print("average position error: ", self.total_position_error / ((len(self.x_hat)) - 1))
            

            print("Total processing time: ", self.total_processing_time)
            print("average processing time per step: ", self.total_processing_time / (len(self.x_hat) - 1))




class KalmanFilter(Estimator):
    """Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    linear unicycle model at the default bearing of pi/4. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive Kalman filter update rule.

    Attributes:
    ----------
        phid : float
            Default bearing of the turtlebot fixed at pi / 4.

    Example
    ----------
    To run the Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=kalman_filter \
            noise_injection:=true \
            freeze_bearing:=true
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Kalman Filter'
        self.phid = np.pi / 4



        
        # State: [x, y, theta_L, theta_R]
        self.A = np.eye(4)

        self.B = np.array([
            [self.r/2 * np.cos(self.phid), self.r/2 * np.cos(self.phid)],
            [self.r/2 * np.sin(self.phid), self.r/2 * np.sin(self.phid)],
            [1, 0],
            [0, 1]
        ])

        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Tuning Q, R, and P0
        self.Q = np.array([
            [0.001, 0, 0, 0],
            [0, 0.001, 0, 0],
            [0, 0, 0.001, 0],
            [0, 0, 0, 0.001]
        ])

        self.R = np.array([
            [0.01, 0],
            [0, 0.01]
        ])

        self.P = np.array([
            [0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1]
        ])

        

    # noinspection DuplicatedCode
    # noinspection PyPep8Naming    
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:


            start_time = rospy.Time.now()  # Get the current time

            last_state = self.x_hat[-1]
            current_time = self.x[-1][0]
            last_time = last_state[0]
            
            # Latest y (measurement) and u (input)
            latest_y = None
            for y_data in reversed(self.y):
                if y_data[0] <= current_time:
                    latest_y = y_data
                    break
            
            latest_u = None
            for u_data in reversed(self.u):
                if u_data[0] <= current_time:
                    latest_u = u_data
                    break
            
            if latest_y is None or latest_u is None:
                self.x_hat.append(last_state)
                return
            
            # x, dt, u, and y
            x_hat = np.array([last_state[2], last_state[3], last_state[4], last_state[5]])
            dt = current_time - last_time
            u = np.array([latest_u[1], latest_u[2]])
            y = np.array([latest_y[1], latest_y[2]])
            
            # Kalman Filter prediction step
            x_pred = self.A.dot(x_hat) + self.B.dot(u) * dt  # State extrapolation
            P_pred = self.A.dot(self.P).dot(self.A.T) + self.Q  # Covariance extrapolation
            
            S = self.C.dot(P_pred).dot(self.C.T) + self.R
            K = P_pred.dot(self.C.T).dot(np.linalg.inv(S))  # Kalman gain
            
            x_new = x_pred + K.dot(y - self.C.dot(x_pred))  # State Update
            self.P = (np.eye(4) - K.dot(self.C)).dot(P_pred)  # Covariance Update
            
            # New state estimate
            new_state = [current_time, self.phid, x_new[0], x_new[1], x_new[2], x_new[3]]
            self.x_hat.append(new_state)

            #calculate how much time it takes to process one step
            total_time_per_step = (rospy.Time.now() - start_time).to_sec()
            self.total_processing_time += total_time_per_step

            #calculate the difference between true state and estimated states in meters (only using x and y)
            true_x = self.x[-1][2]
            true_y = self.x[-1][3]


            print(len(self.x_hat))
            position_error = np.sqrt((true_x -  x_new[0]) ** 2 + (true_y - x_new[1]) ** 2)
            self.total_position_error += position_error
            print("distance: ", position_error)
            print("average position error: ", self.total_position_error / ((len(self.x_hat)) - 1))
            

            print("Total processing time: ", self.total_processing_time)
            print("average processing time per step: ", self.total_processing_time / (len(self.x_hat) - 1))


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

    Example
    ----------
    To run the extended Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=extended_kalman_filter \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Extended Kalman Filter'
        self.landmark = (0.5, 0.5)

        self.Q = np.array([
            [0.1, 0, 0, 0, 0, 0],
            [0, 0.001, 0, 0, 0, 0],
            [0, 0, 0.001, 0, 0, 0],
            [0, 0, 0, 0.001, 0, 0],
            [0, 0, 0, 0, 0.001, 0],
            [0, 0, 0, 0, 0, 0.001]
        ])
        
        self.R = np.array([
            [8, 0],
            [0, 10]
        ])
        
        self.P = np.array([
            [0.1, 0, 0, 0, 0, 0],
            [0, 0.001, 0, 0, 0, 0],
            [0, 0, 0.001, 0, 0, 0],
            [0, 0, 0, 0.001, 0, 0],
            [0, 0, 0, 0, 0.001, 0],
            [0, 0, 0, 0, 0, 0.001]
        ])


    # noinspection DuplicatedCode
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # Get the latest state estimate and time
            last_state = self.x_hat[-1]
            current_time = self.x[-1][0]
            last_time = last_state[0]
            dt = current_time - last_time
            
            # Get latest measurement
            latest_y = None
            for y_data in reversed(self.y):
                if y_data[0] <= current_time:
                    latest_y = y_data
                    break
            
            # Get latest input
            latest_u = None
            for u_data in reversed(self.u):
                if u_data[0] <= current_time:
                    latest_u = u_data
                    break
            
            if latest_y is None or latest_u is None:
                self.x_hat.append(last_state)
                return
            
            x_hat = np.array(last_state)
            u = np.array(latest_u)
            y = np.array(latest_y[1:3])
            
            # State extrapolation
            x_pred = self.g(x_hat, u, dt)
            
            # dynamics linearization
            A = self.A(x_hat, u, dt)
            
            # covariance extrapolation
            P_pred = A @ self.P @ A.T + self.Q
            
            # Measurement linearization
            C = self.C(x_pred)
            
            # Kalman gain
            innovation = y - self.h(x_pred)
            K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + self.R)
            
            # state update
            x_updated = x_pred + K @ innovation
            
            # covariance update
            I = np.eye(6)
            self.P = (I - K @ C) @ P_pred
            
            # new state estimate
            self.x_hat.append(x_updated)
        
    def g(self, state, u, dt):
        """
        Dynamics model function for the unicycle model.
        
        Parameters:
        state : current state [time, phi, x, y, theta_L, theta_R]
        u : control input [u_L, u_R]
        dt : time step
        
        Returns:
        The predicted next state using forward Euler integration
        """
        # state components
        phi = state[1]
        x_pos = state[2]
        y_pos = state[3]
        theta_L = state[4]
        theta_R = state[5]
        
        # inputs
        u_L = u[1]
        u_R = u[2]
        
        # unicycle
        new_phi = phi + (self.r / (2 * self.d)) * (u_R - u_L) * dt
        new_x = x_pos + (self.r / 2) * (u_L + u_R) * np.cos(phi) * dt
        new_y = y_pos + (self.r / 2) * (u_L + u_R) * np.sin(phi) * dt
        new_theta_L = theta_L + u_L * dt
        new_theta_R = theta_R + u_R * dt
        
        return np.array([state[0] + dt, new_phi, new_x, new_y, new_theta_L, new_theta_R])

    def h(self, state):
        """
        Measurement model function.
        
        Parameters:
        state : current state [time, phi, x, y, theta_L, theta_R]
        
        Returns:
        The predicted measurement [distance to landmark, relative bearing]
        """
        x_pos = state[2]
        y_pos = state[3]
        phi = state[1]
        
        dx = self.landmark[0] - x_pos
        dy = self.landmark[1] - y_pos
        
        distance = np.sqrt(dx**2 + dy**2)
        dphi = np.arctan2(dy, dx) - phi
            
        return np.array([distance, dphi])
    
    def A(self, state, u, dt):
        """
        Compute the Jacobian of the dynamics model with respect to the state.
        
        Parameters:
        state : current state [time, phi, x, y, theta_L, theta_R]
        u : control input [u_L, u_R]
        dt : time step
        
        Returns:
        The linearized dynamics matrix A
        """
        phi = state[1]
        u1 = u[1]
        u2 = u[2]

        A = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, -(self.r / 2) * (u1 + u2) * np.sin(phi) * dt, 1, 0, 0, 0],
                      [0, (self.r / 2) * (u1 + u2) * np.cos(phi) * dt, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        
        return A
    
    def C(self, state):
        """
        Compute the Jacobian of the measurement model with respect to the state.
        
        Parameters:
        state : current state [time, phi, x, y, theta_L, theta_R]
        
        Returns:
        The linearized measurement matrix C
        """
        # Extract robot position and orientation
        x_pos = state[2]
        y_pos = state[3]
        phi = state[1]
        
        # Calculate relative position to landmark
        dx = self.landmark[0] - x_pos
        dy = self.landmark[1] - y_pos
        
        # Calculate distance to landmark
        dist_squared = dx**2 + dy**2
        distance = np.sqrt(dist_squared)
        
        if dist_squared < 1e-10:
            C = np.array([[0, 0, -1, 0, 0, 0, 0],
                        [0, -1, 0, 0, 0, 0, 0]])
        else:
            C = np.array([[0, 0, -dx / distance, -dy / distance, 0, 0],
                        [0, -1, dy / dist_squared, -dx / dist_squared, 0, 0]])
        
        return C

