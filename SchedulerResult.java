import java.util.*;

/**
 * Result container returned by the scheduler.
 */
public class SchedulerResult {
    public int[] assignment;     // assignment[i] = slot index (0-indexed), or -1 if unassigned
    public double penalty;
    public int runtimeMs;
    public boolean feasible;
    public String violationReason;

    public SchedulerResult(int n) {
        this.assignment = new int[n];
        Arrays.fill(this.assignment, -1);
    }

    /**
     * Convert to a Map suitable for JSON serialization.
     * Uses task-indexed keys (T0, T1, ...) for the assignment dict.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        Map<String, Integer> assignMap = new LinkedHashMap<>();
        for (int i = 0; i < assignment.length; i++) {
            assignMap.put("T" + i, assignment[i]);
        }
        map.put("assignment", assignMap);
        map.put("penalty", penalty);
        map.put("runtime_ms", runtimeMs);
        map.put("feasible", feasible);
        if (violationReason != null) {
            map.put("violation_reason", violationReason);
        }
        return map;
    }
}
