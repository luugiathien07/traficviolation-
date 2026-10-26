package vn.tvd.track;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Minimum-cost assignment with a per-pair cost limit, like lap.lapjv(extend_cost=True, cost_limit=thresh):
 * pairs costing more than the limit are left unmatched. Hungarian algorithm on the cost matrix extended
 * with "unmatched" slots that cost limit/2 each.
 */
public final class LinearAssignment {
    public record Result(List<int[]> matches, List<Integer> unmatchedA, List<Integer> unmatchedB) {
    }

    private LinearAssignment() {
    }

    public static Result solve(double[][] cost, int rows, int cols, double limit) {
        List<int[]> matches = new ArrayList<>();
        List<Integer> ua = new ArrayList<>(), ub = new ArrayList<>();
        if (rows == 0 || cols == 0) {
            for (int i = 0; i < rows; i++) {
                ua.add(i);
            }
            for (int j = 0; j < cols; j++) {
                ub.add(j);
            }
            return new Result(matches, ua, ub);
        }
        int n = rows + cols;
        double[][] c = new double[n][n];
        double half = limit / 2;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i < rows && j < cols) {
                    c[i][j] = cost[i][j];
                } else if (i < rows || j < cols) {
                    c[i][j] = half;
                } else {
                    c[i][j] = 0;
                }
            }
        }
        int[] rowOf = hungarian(c);  // rowOf[col] = row
        int[] colOf = new int[rows];
        Arrays.fill(colOf, -1);
        boolean[] colMatched = new boolean[cols];
        for (int j = 0; j < cols; j++) {
            int i = rowOf[j];
            if (i < rows && cost[i][j] <= limit) {
                colOf[i] = j;
                colMatched[j] = true;
                matches.add(new int[] {i, j});
            }
        }
        for (int i = 0; i < rows; i++) {
            if (colOf[i] < 0) {
                ua.add(i);
            }
        }
        for (int j = 0; j < cols; j++) {
            if (!colMatched[j]) {
                ub.add(j);
            }
        }
        return new Result(matches, ua, ub);
    }

    /** Square Hungarian (potentials, O(n^3)). Returns the row assigned to each column. */
    static int[] hungarian(double[][] a) {
        int n = a.length;
        double[] u = new double[n + 1], v = new double[n + 1];
        int[] p = new int[n + 1], way = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            p[0] = i;
            int j0 = 0;
            double[] minv = new double[n + 1];
            Arrays.fill(minv, Double.POSITIVE_INFINITY);
            boolean[] used = new boolean[n + 1];
            do {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                double delta = Double.POSITIVE_INFINITY;
                for (int j = 1; j <= n; j++) {
                    if (!used[j]) {
                        double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }
                for (int j = 0; j <= n; j++) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }
        int[] rowOf = new int[n];
        for (int j = 1; j <= n; j++) {
            rowOf[j - 1] = p[j] - 1;
        }
        return rowOf;
    }
}
