# trajectory_estimator.py
import numpy as np
from typing import List, Tuple

from python_utils.plotter import Plotter

global_plotter = Plotter()

class DataSimulation:
    def __init__(self, gravity: np.ndarray = np.array([0.0, 0.0, -9.81]),):
        self.gravity = gravity
    def simulate_data(self,
                      p0: np.ndarray,
                      v0: np.ndarray,
                      times: np.ndarray,
                      noise_std: float = 0.05) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generate noisy observations along a true parabola defined by p0, v0.
        returns (points, times)
        """
        points = []
        for t in times:
            true_pt = p0 + v0 * t + 0.5 * self.gravity * t**2
            noisy_pt = true_pt + np.random.normal(scale=noise_std, size=3)
            points.append(noisy_pt)
        return points, times.tolist()

class NewtonLawTrajectoryPrediction:
    """
    Class to estimate a 3D parabolic trajectory from noisy observations.
    Includes methods to simulate data, fit via RANSAC, refine by least squares,
    and predict future trajectory points.
    """
    def __init__(self,
                 gravity: np.ndarray = np.array([0.0, 0.0, -9.81]),
                 ransac_threshold: float = 0.05,
                 ransac_iters: int = 100):
        self.gravity = gravity
        self.threshold = ransac_threshold
        self.max_iters = ransac_iters

        self._p0 = None
        self._v0 = None
    
    # Step 1: Tìm phương trình Parabol dựa vào 2 điểm dữ liệu
    def _fit_parabola_two_points(self, p1: np.ndarray, t1: float,
                                 p2: np.ndarray, t2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit parabola p(t)=p0+v0*t+0.5*g*t^2 from two observations
        """
        A = np.zeros((6,6))
        b = np.zeros(6)
        # Construct linear system for p0 (3) and v0 (3)
        # For each point: p_i = p0 + v0 * t_i + 0.5*g*t_i^2
        # => p_i - 0.5*g*t_i^2 = [I, t_i*I] [p0; v0]
        def build_row(pt, ti, row_offset):
            y = pt - 0.5 * self.gravity * ti**2
            A[row_offset:row_offset+3, 0:3] = np.eye(3)
            A[row_offset:row_offset+3, 3:6] = np.eye(3) * ti
            b[row_offset:row_offset+3] = y
            
        build_row(p1, t1, 0)
        build_row(p2, t2, 3)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return x[:3], x[3:] # p0, v0

    # Step 2: Tìm các điểm dữ liệu gần nhất với parabol (Tìm inliers)
    def fit_ransac(self,
                   points: List[np.ndarray],
                   times: List[float]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Perform RANSAC to find the parabola with most inliers.
        returns p0, v0, inlier_indices
        """
        best_inliers = []
        best_p0, best_v0 = None, None
        n = len(points)
        for _ in range(self.max_iters):
            i1, i2 = np.random.choice(n, 2, replace=False)
            p0_candidate, v0_candidate = self._fit_parabola_two_points(
                points[i1], times[i1], points[i2], times[i2]
            )
            inliers = []
            for i, (p, t) in enumerate(zip(points, times)):
                pred = p0_candidate + v0_candidate * t + 0.5 * self.gravity * t**2
                if np.linalg.norm(p - pred) <= self.threshold:
                    inliers.append(i)
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_p0, best_v0 = p0_candidate, v0_candidate
        return best_p0, best_v0, best_inliers

    # Step 3: Tính toán lại p0, v0 từ cụm các điểm inliers được tìm ra ở step 4 dựa vào least squares
    def refine_least_squares(self,
                             points: List[np.ndarray],
                             times: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine parabola parameters using all inliers via least squares.
        returns refined p0, v0
        """
        N = len(points)             # là số lượng điểm inliers
        Y = np.zeros(3 * N)         # là vector chứa các điểm inliers, mỗi điểm có 3 chiều
        X = np.zeros((3 * N, 6))    # là ma trận kích thước (3N)x6: 6 chiều cho p0 và v0

        # Setup ma trận X và vector Y bằng cách duyệt qua từng điểm inliers
        for i, (p, t) in enumerate(zip(points, times)):
            y = p - 0.5 * self.gravity * t**2
            Y[3*i:3*i+3] = y
            X[3*i:3*i+3, 0:3] = np.eye(3)           # np.eye(3) là ma trận đơn vị 3x3
            X[3*i:3*i+3, 3:6] = np.eye(3) * t
        theta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return theta[:3], theta[3:] # p0, v0

    def update_model(self, points_np, times_np):
        p0, v0, inliers = self.fit_ransac(points_np, times_np)
        p0_refined, v0_refined = self.refine_least_squares(points_np[inliers], times_np[inliers])
        self._p0 = p0_refined
        self._v0 = v0_refined

    def predict(self, num_points: int, dt: float) -> List[np.ndarray]:
        """
        Predict future trajectory points at uniform time intervals.
        """
        return np.array([self._p0 + self._v0 * t + 0.5 * self.gravity * t**2
                for t in np.linspace(0, dt*(num_points-1), num_points)])

def main():
    dsim = DataSimulation()
    points, times = dsim.simulate_data(
        p0=np.array([0.0, 0.0, 0.0]),
        v0=np.array([10.0, 1.0, 5.0]),
        times=np.linspace(0, 2, 20),
        noise_std=0.1
    )
    points = np.array(points)
    times = np.array(times)

    ntp = NewtonLawTrajectoryPrediction()
    # p0, v0, inliers = ntp.fit_ransac(points, times)
    # p0_refined, v0_refined = ntp.refine_least_squares(points[inliers], times[inliers])

    ntp.update_model(points, times)
    predicted_points = ntp.predict(num_points=30, dt=0.1)
    print('points shape:', points.shape)
    print('predicted_points shape:', predicted_points.shape)
    global_plotter.plot_trajectory_dataset_plotly(trajectories=[points, predicted_points], title='Newton law Trajectory prediction')

if __name__ == "__main__":
    main()


    

