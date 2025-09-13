import numpy as np

class NewtonianTrajectoryPrediction:
    """
    Ước tính parabol 3D dạng p(t) = ½ g t² + v t + p bằng RANSAC + Least-Squares
    Hệ tọa độ z-up, gravity kéo xuống theo -z.
    """
    def __init__(
        self,
        gravity: np.ndarray = np.array([0.0, 0.0, -9.81]),
        confidence: float = 0.999,
        num_max_ransac_iter: int = 100,
        inlier_threshold: float = 0.1,
        seed=42
    ):
        self.gravity = gravity
        self.confidence = confidence
        self.num_max_iter = num_max_ransac_iter
        self.inlier_threshold = inlier_threshold
        np.random.seed(seed)  # Set random seed for reproducibility

    def minimal_solution(self, points: np.ndarray, times: np.ndarray):
        '''
        Giải hệ phương trình, tìm p_init, v_init của parabol đi qua 2 điểm (minimal case).
        Note: chỉ có 1 nghiệm duy nhất vì parabol được constrainted bởi vector gravity g
        '''
        t0, t1 = times
        p0, p1 = points
        p0_gc = p0 - 0.5 * self.gravity * t0**2
        p1_gc = p1 - 0.5 * self.gravity * t1**2
        v = (p1_gc - p0_gc) / (t1 - t0)
        p = p0_gc - v * t0
        return p, v

    def least_squares_solution(self, points: np.ndarray, times: np.ndarray):
        P = points - 0.5 * self.gravity.reshape((1,3)) * times.reshape((-1,1))**2
        n    = len(times)
        t    = times.sum()
        t2   = (times**2).sum()
        p_sum  = P.sum(axis=0)
        tp_sum = (P * times.reshape((-1,1))).sum(axis=0)
        denom = t**2 - n * t2
        p_opt = (t * tp_sum - p_sum * t2) / denom
        v_opt = (t * p_sum   - n     * tp_sum) / denom
        return p_opt, v_opt

    def compute_inliers(self, points: np.ndarray, times: np.ndarray, p: np.ndarray, v: np.ndarray):
        P    = points - 0.5 * self.gravity.reshape((1,3)) * times.reshape((-1,1))**2
        pred = p.reshape((1,3)) + v.reshape((1,3)) * times.reshape((-1,1))
        errs = P - pred
        return (errs**2).sum(axis=1) < self.inlier_threshold**2

    def _ransac_iters(self, outlier_prob: float):
        return int(np.log(1-self.confidence) / np.log(1-(1-outlier_prob)**2))

    def fit_ransac(self, points_3d: np.ndarray, times: np.ndarray, first_point_constraint=False):
        """
        Thực hiện RANSAC + Least-Squares để tìm p_init, v_init.
        first_point_constraint: nếu True thì ép parabol luôn đi qua điểm đầu tiên (points_3d[0])
        Trả về (p_init, v_init, inlier_mask)
        """
        best_inliers = np.zeros(len(points_3d), dtype=bool)
        max_inliers  = 0
        num_iter = self.num_max_iter

        for _ in range(num_iter):
            # chọn random 2 điểm để tính p, v của parabol đi qua 2 điểm đó
            if first_point_constraint:
                j = np.random.choice(len(points_3d) - 1) + 1   # chọn ngẫu nhiên từ 1..len-1
                idx = np.array([0, j])                        # luôn chứa 0
            else:
                idx = np.random.choice(len(points_3d), size=2, replace=False)

            p0, v0 = self.minimal_solution(points_3d[idx], times[idx])

            inliers = self.compute_inliers(points_3d, times, p0, v0)
            n_in = inliers.sum()
            # update inlier list if number of inliers is the most so far
            if n_in > max_inliers:
                max_inliers  = n_in
                best_inliers = inliers

            outlier_prob = np.clip(1 - n_in / len(points_3d), 1e-6, 1-1e-6)
            num_iter = min(self._ransac_iters(outlier_prob), self.num_max_iter)

        # tinh chỉnh cuối cùng trên tập inliers tốt nhất
        p_opt, v_opt = self.least_squares_solution(points_3d[best_inliers], times[best_inliers])
        return p_opt, v_opt, best_inliers

    def update_model(self, points_np, times_np, first_point_constraint=False):
        p0, v0, inliers = self.fit_ransac(points_np, times_np, first_point_constraint=first_point_constraint)
        return p0, v0

    def predict_points_at_timestamps(self, t_arr: np.ndarray, p0: np.ndarray, v0: np.ndarray) -> np.ndarray:
        """
        Predict point(s) at time t.
        args:
          t: float or 1D-array of floats
        returns:
          nếu t là scalar  -> array shape (d,)
          nếu t là array   -> array shape (len(t), d)
        """
        t_arr = np.asarray(t_arr)                      # chuyển về ndarray
        # thêm chiều để broadcast: t_arr[..., None] có shape (n,1) hoặc ()->[ ]
        pts = p0 + v0 * t_arr[..., None] + 0.5 * self.gravity * (t_arr[..., None]**2)
        return pts


def main():
    from python_utils.plotter import Plotter
    util_plotter = Plotter()

    ## ----- 1. Tạo quỹ đạo mẫu trong hệ z-up, dấu dương hướng lên
    g_world = np.array([0.0, 0.0, -9.81])    # gravity vector in z-up (m/s^2)
    p0_world = np.array([0.0, 0.0, 1.0])     # Vị trí ban đầu (x, y, z)
    v0_world = np.array([2.0, 0.0, 5.0])     # Vận tốc ban đầu (vx, vy, vz)

    #   Thời điểm lấy mẫu
    times = np.linspace(0, 0.5, 20)          # 20 điểm từ t=0 đến t=1.5s

    #   Tính quỹ đạo lý thuyết trong hệ world, hệ z-up, nhìn g_world thì biết
    points_3d = 0.5 * g_world * times[:, None]**2 + v0_world * times[:, None] + p0_world
    noise = np.random.normal(0.0, 0.01, points_3d.shape)  # Thêm nhiễu Gaussian
    # points3d_fit += noise
    # points_3d = points_3d + noise
    # util_plotter.plot_trajectory_dataset_matplotlib(samples=[points_3d], title='Original Parabola Trajectory')   # OK



    ## ----- 2. fit
    ntp = NewtonianTrajectoryPrediction(
        gravity=g_world,
        confidence=0.999,
        num_max_ransac_iter=100,
        inlier_threshold=0.1
    )
    p_init, v_init, inlier_mask = ntp.fit_ransac(
        points_3d,
        times)
    t_query = np.linspace(0, 1, 39)        # 100 thời điểm nội suy
    pred_seq = ntp.predict_points_at_timestamps(t_query, p_init, v_init)
    util_plotter.plot_predictions_plotly(inputs=[points_3d], labels=[points_3d], predictions=[pred_seq])
    print(f'Predicted p: ', np.array2string(p_init, precision=3))
    print(f'Predicted v: ', np.array2string(v_init, precision=3))



if __name__ == "__main__":
    main()
