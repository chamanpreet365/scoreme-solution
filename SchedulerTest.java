import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

/**
 * Unit tests for the PWDRAB scheduler.
 * Covers: all-conflict graph, zero-capacity slot, tight SLA windows,
 * single-task instance, and penalty monotonicity.
 */
class SchedulerTest {

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    private InstanceData buildInstance(
            int n, int K,
            int[][] conflicts,
            double[][] resources,
            double[][] capacities,
            int[][] windows,
            double[] weights) {

        InstanceData inst = new InstanceData();
        inst.n = n;
        inst.K = K;
        inst.d = 4;
        inst.tasks = new String[n];
        for (int i = 0; i < n; i++) inst.tasks[i] = "T" + i;
        inst.conflicts = conflicts;
        inst.resources = resources;
        inst.capacities = capacities;
        inst.windows = windows;
        inst.weights = weights;
        inst.conflictAdj = new Set[n];
        for (int i = 0; i < n; i++) inst.conflictAdj[i] = new HashSet<>();
        for (int[] e : conflicts) {
            inst.conflictAdj[e[0]].add(e[1]);
            inst.conflictAdj[e[1]].add(e[0]);
        }
        return inst;
    }

    private double[][] uniformResources(int n, double cpu, double ram, double gpu, double net) {
        double[][] r = new double[n][4];
        for (double[] row : r) { row[0]=cpu; row[1]=ram; row[2]=gpu; row[3]=net; }
        return r;
    }

    private double[][] uniformCapacities(int K, double cpu, double ram, double gpu, double net) {
        double[][] c = new double[K][4];
        for (double[] row : c) { row[0]=cpu; row[1]=ram; row[2]=gpu; row[3]=net; }
        return c;
    }

    private int[][] fullWindows(int n, int K) {
        int[][] w = new int[n][2];
        for (int[] row : w) { row[0]=0; row[1]=K-1; }
        return w;
    }

    private double[] uniformWeights(int n, double v) {
        double[] w = new double[n]; Arrays.fill(w, v); return w;
    }

    // -----------------------------------------------------------------------
    // Test 1: Single-task instance — should always produce a feasible solution
    // -----------------------------------------------------------------------
    @Test
    void testSingleTask() {
        InstanceData inst = buildInstance(
            1, 3,
            new int[0][2],
            uniformResources(1, 1, 1, 0, 0),
            uniformCapacities(3, 32, 128, 8, 6),
            new int[][]{{0, 2}},
            uniformWeights(1, 5.0)
        );
        Scheduler s = new Scheduler(inst);
        SchedulerResult r = s.solve();
        assertTrue(r.feasible, "Single task must always be feasibly assigned");
        assertEquals(0, r.assignment[0], "Should be placed in earliest slot (slot 0) to minimise penalty");
    }

    // -----------------------------------------------------------------------
    // Test 2: All-conflict graph with K ≥ n — should be feasible (each task in its own slot)
    // -----------------------------------------------------------------------
    @Test
    void testAllConflictFeasible() {
        int n = 4; int K = 4;
        // Build complete conflict graph
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++) edges.add(new int[]{i,j});
        InstanceData inst = buildInstance(
            n, K,
            edges.toArray(new int[0][]),
            uniformResources(n, 1, 1, 0, 0),
            uniformCapacities(K, 32, 128, 8, 6),
            fullWindows(n, K),
            uniformWeights(n, 1.0)
        );
        Scheduler s = new Scheduler(inst);
        SchedulerResult r = s.solve();
        assertTrue(r.feasible, "Complete conflict graph with K=n must be feasible");
        // Verify no two tasks share a slot
        Set<Integer> usedSlots = new HashSet<>();
        for (int slot : r.assignment) {
            assertFalse(usedSlots.contains(slot), "All tasks must be in distinct slots");
            usedSlots.add(slot);
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: All-conflict graph with K < n — should be infeasible
    // -----------------------------------------------------------------------
    @Test
    void testAllConflictInfeasible() {
        int n = 5; int K = 3;
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++) edges.add(new int[]{i,j});
        InstanceData inst = buildInstance(
            n, K,
            edges.toArray(new int[0][]),
            uniformResources(n, 1, 1, 0, 0),
            uniformCapacities(K, 32, 128, 8, 6),
            fullWindows(n, K),
            uniformWeights(n, 1.0)
        );
        Scheduler s = new Scheduler(inst);
        SchedulerResult r = s.solve();
        assertFalse(r.feasible, "Complete graph with n=5, K=3 is infeasible (chromatic number = 5 > K)");
        assertNotNull(r.violationReason, "Infeasible result must include violation reason");
    }

    // -----------------------------------------------------------------------
    // Test 4: Zero-capacity slot — tasks should not be placed in it
    // -----------------------------------------------------------------------
    @Test
    void testZeroCapacitySlot() {
        int n = 2; int K = 3;
        // Slot 0 has zero CPU capacity → no task can go there
        double[][] caps = {
            {0.0, 128, 8, 6},  // slot 0: zero CPU — unusable
            {32,  128, 8, 6},
            {32,  128, 8, 6}
        };
        InstanceData inst = buildInstance(
            n, K,
            new int[0][2],
            uniformResources(n, 1, 1, 0, 0),
            caps,
            fullWindows(n, K),
            uniformWeights(n, 1.0)
        );
        Scheduler s = new Scheduler(inst);
        SchedulerResult r = s.solve();
        assertTrue(r.feasible, "Should be feasible using slots 1 and 2");
        for (int slot : r.assignment) {
            assertNotEquals(0, slot, "No task should be assigned to zero-capacity slot 0");
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: Tight SLA windows — each task can only go in one specific slot
    // -----------------------------------------------------------------------
    @Test
    void testTightSlaWindows() {
        int n = 3; int K = 3;
        // Task i must go in slot i exactly
        int[][] windows = {{0,0}, {1,1}, {2,2}};
        InstanceData inst = buildInstance(
            n, K,
            new int[0][2],
            uniformResources(n, 1, 1, 0, 0),
            uniformCapacities(K, 32, 128, 8, 6),
            windows,
            uniformWeights(n, 1.0)
        );
        Scheduler s = new Scheduler(inst);
        SchedulerResult r = s.solve();
        assertTrue(r.feasible);
        assertEquals(0, r.assignment[0], "T0 must be in slot 0");
        assertEquals(1, r.assignment[1], "T1 must be in slot 1");
        assertEquals(2, r.assignment[2], "T2 must be in slot 2");
    }

    // -----------------------------------------------------------------------
    // Test 6: Tight SLA + conflict forces infeasibility
    // -----------------------------------------------------------------------
    @Test
    void testTightSlaConflictInfeasible() {
        // T0 and T1 conflict AND both must be in slot 0 → infeasible
        int n = 2; int K = 2;
        int[][] windows = {{0,0}, {0,0}};
        InstanceData inst = buildInstance(
            n, K,
            new int[][]{{0,1}},
            uniformResources(n, 1, 1, 0, 0),
            uniformCapacities(K, 32, 128, 8, 6),
            windows,
            uniformWeights(n, 1.0)
        );
        Scheduler s = new Scheduler(inst);
        SchedulerResult r = s.solve();
        assertFalse(r.feasible, "Conflicting tasks with identical single-slot windows must be infeasible");
    }

    // -----------------------------------------------------------------------
    // Test 7: Penalty is monotonically non-negative
    // -----------------------------------------------------------------------
    @Test
    void testPenaltyNonNegative() {
        int n = 6; int K = 4;
        InstanceData inst = buildInstance(
            n, K,
            new int[][]{{0,1},{1,2},{2,3}},
            uniformResources(n, 2, 8, 1, 0.5),
            uniformCapacities(K, 32, 128, 8, 6),
            fullWindows(n, K),
            uniformWeights(n, 3.0)
        );
        Scheduler s = new Scheduler(inst);
        SchedulerResult r = s.solve();
        assertTrue(r.penalty >= 0, "Penalty must always be non-negative");
    }

    // -----------------------------------------------------------------------
    // Test 8: Resource capacity violation is detected
    // -----------------------------------------------------------------------
    @Test
    void testResourceCapacityViolationDetected() {
        // All tasks require 20 CPU, K=1 slot has only 32 CPU → only 1 task can fit
        int n = 3; int K = 1;
        InstanceData inst = buildInstance(
            n, K,
            new int[0][2],
            uniformResources(n, 20, 1, 0, 0),
            uniformCapacities(K, 32, 128, 8, 6),
            fullWindows(n, K),
            uniformWeights(n, 1.0)
        );
        Scheduler s = new Scheduler(inst);
        SchedulerResult r = s.solve();
        assertFalse(r.feasible, "3 tasks each needing 20 CPU cannot fit in a 32-CPU single slot");
    }
}
