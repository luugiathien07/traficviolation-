package vn.tvd.track;

/**
 * Constant-velocity Kalman filter on (centre x, centre y, aspect ratio, height), as in ByteTrack /
 * Ultralytics KalmanFilterXYAH. State is 8-dim: position and velocity of those four values.
 */
public final class KalmanFilter {
    private static final double W_POS = 1.0 / 20, W_VEL = 1.0 / 160;

    public double[][] initiate(double[] m, double[] meanOut) {
        System.arraycopy(m, 0, meanOut, 0, 4);
        for (int i = 4; i < 8; i++) {
            meanOut[i] = 0;
        }
        double h = m[3];
        double[] std = {2 * W_POS * h, 2 * W_POS * h, 1e-2, 2 * W_POS * h,
            10 * W_VEL * h, 10 * W_VEL * h, 1e-5, 10 * W_VEL * h};
        double[][] cov = new double[8][8];
        for (int i = 0; i < 8; i++) {
            cov[i][i] = std[i] * std[i];
        }
        return cov;
    }

    /** In place: mean, and returns the new covariance. */
    public double[][] predict(double[] mean, double[][] cov) {
        double h = mean[3];
        double[] std = {W_POS * h, W_POS * h, 1e-2, W_POS * h, W_VEL * h, W_VEL * h, 1e-5, W_VEL * h};
        for (int i = 0; i < 4; i++) {
            mean[i] += mean[i + 4];
        }
        // F P F^T with F = [[I, I], [0, I]]
        double[][] fp = new double[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                fp[i][j] = cov[i][j] + (i < 4 ? cov[i + 4][j] : 0);
            }
        }
        double[][] out = new double[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                out[i][j] = fp[i][j] + (j < 4 ? fp[i][j + 4] : 0);
            }
            out[i][i] += std[i] * std[i];
        }
        return out;
    }

    /** In place: mean, and returns the new covariance. */
    public double[][] update(double[] mean, double[][] cov, double[] z) {
        double h = mean[3];
        double[] std = {W_POS * h, W_POS * h, 1e-1, W_POS * h};
        // S = H P H^T + R (H picks the first 4 states)
        double[][] s = new double[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                s[i][j] = cov[i][j];
            }
            s[i][i] += std[i] * std[i];
        }
        double[][] sInv = invert4(s);
        // K = P H^T S^-1  (8x4)
        double[][] k = new double[8][4];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                double v = 0;
                for (int l = 0; l < 4; l++) {
                    v += cov[i][l] * sInv[l][j];
                }
                k[i][j] = v;
            }
        }
        double[] y = new double[4];
        for (int i = 0; i < 4; i++) {
            y[i] = z[i] - mean[i];
        }
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                mean[i] += k[i][j] * y[j];
            }
        }
        // P - K S K^T  ==  P - K H P
        double[][] out = new double[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                double v = 0;
                for (int l = 0; l < 4; l++) {
                    v += k[i][l] * cov[l][j];
                }
                out[i][j] = cov[i][j] - v;
            }
        }
        return out;
    }

    static double[][] invert4(double[][] m) {
        int n = 4;
        double[][] a = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, a[i], 0, n);
            a[i][n + i] = 1;
        }
        for (int c = 0; c < n; c++) {
            int piv = c;
            for (int r = c + 1; r < n; r++) {
                if (Math.abs(a[r][c]) > Math.abs(a[piv][c])) {
                    piv = r;
                }
            }
            double[] tmp = a[c];
            a[c] = a[piv];
            a[piv] = tmp;
            double d = a[c][c];
            for (int j = 0; j < 2 * n; j++) {
                a[c][j] /= d;
            }
            for (int r = 0; r < n; r++) {
                if (r != c) {
                    double f = a[r][c];
                    for (int j = 0; j < 2 * n; j++) {
                        a[r][j] -= f * a[c][j];
                    }
                }
            }
        }
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], n, inv[i], 0, n);
        }
        return inv;
    }
}
