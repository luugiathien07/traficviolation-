package vn.tvd.track;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Test;

class LinearAssignmentTest {
    /** Brute force over all partial matchings: unmatched rows/cols cost limit/2 each, pairs above the limit not allowed. */
    static double best(double[][] c, int row, boolean[] usedCol, double limit) {
        if (row == c.length) {
            double s = 0;
            for (boolean u : usedCol) {
                s += u ? 0 : limit / 2;
            }
            return s;
        }
        double b = limit / 2 + best(c, row + 1, usedCol, limit);
        for (int j = 0; j < usedCol.length; j++) {
            if (!usedCol[j] && c[row][j] <= limit) {
                usedCol[j] = true;
                b = Math.min(b, c[row][j] + best(c, row + 1, usedCol, limit));
                usedCol[j] = false;
            }
        }
        return b;
    }

    @Test
    void matchesBruteForceOnRandomMatrices() {
        Random rnd = new Random(7);
        for (int trial = 0; trial < 300; trial++) {
            int n = 1 + rnd.nextInt(5), m = 1 + rnd.nextInt(5);
            double[][] c = new double[n][m];
            for (double[] row : c) {
                for (int j = 0; j < m; j++) {
                    row[j] = rnd.nextDouble();
                }
            }
            double limit = 0.8;
            LinearAssignment.Result r = LinearAssignment.solve(c, n, m, limit);
            double total = (r.unmatchedA().size() + r.unmatchedB().size()) * limit / 2;
            for (int[] p : r.matches()) {
                total += c[p[0]][p[1]];
            }
            assertEquals(best(c, 0, new boolean[m], limit), total, 1e-9, "trial " + trial);
        }
    }
}
