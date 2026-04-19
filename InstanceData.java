import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.util.*;

/**
 * Holds the parsed input instance for the scheduling problem.
 * Fields mirror the JSON format produced by generate_instance.py.
 */
public class InstanceData {
    public int n;             // number of tasks
    public int K;             // number of slots
    public int d = 4;         // resource dimensions (CPU, RAM, GPU, Network)

    public String[] tasks;            // task names T0..T_{n-1}
    public int[][] conflicts;         // list of [i,j] conflict pairs
    public double[][] resources;      // resources[i][dim] = requirement
    public double[][] capacities;     // capacities[slot][dim]
    public int[][] windows;           // windows[i] = [lo, hi] (inclusive, 0-indexed)
    public double[] weights;          // priority weights

    // Derived: adjacency set per task
    public Set<Integer>[] conflictAdj;

    @SuppressWarnings("unchecked")
    public static InstanceData fromFile(String path, ObjectMapper mapper) throws Exception {
        Map<String, Object> raw = mapper.readValue(new File(path), Map.class);

        InstanceData inst = new InstanceData();
        List<String> taskList = (List<String>) raw.get("tasks");
        inst.n = taskList.size();
        inst.K = (int) raw.get("K");
        inst.tasks = taskList.toArray(new String[0]);

        // Conflicts
        List<List<Integer>> confList = (List<List<Integer>>) raw.get("conflicts");
        inst.conflicts = new int[confList.size()][2];
        for (int i = 0; i < confList.size(); i++) {
            inst.conflicts[i][0] = confList.get(i).get(0);
            inst.conflicts[i][1] = confList.get(i).get(1);
        }

        // Resources
        List<List<Number>> resList = (List<List<Number>>) raw.get("resources");
        inst.resources = new double[inst.n][inst.d];
        for (int i = 0; i < inst.n; i++) {
            for (int j = 0; j < inst.d; j++) {
                inst.resources[i][j] = resList.get(i).get(j).doubleValue();
            }
        }

        // Capacities
        List<List<Number>> capList = (List<List<Number>>) raw.get("capacities");
        inst.capacities = new double[inst.K][inst.d];
        for (int s = 0; s < inst.K; s++) {
            for (int j = 0; j < inst.d; j++) {
                inst.capacities[s][j] = capList.get(s).get(j).doubleValue();
            }
        }

        // Windows
        List<List<Integer>> winList = (List<List<Integer>>) raw.get("windows");
        inst.windows = new int[inst.n][2];
        for (int i = 0; i < inst.n; i++) {
            inst.windows[i][0] = winList.get(i).get(0);
            inst.windows[i][1] = winList.get(i).get(1);
        }

        // Weights
        List<Number> wList = (List<Number>) raw.get("weights");
        inst.weights = new double[inst.n];
        for (int i = 0; i < inst.n; i++) {
            inst.weights[i] = wList.get(i).doubleValue();
        }

        // Build adjacency sets
        inst.conflictAdj = new Set[inst.n];
        for (int i = 0; i < inst.n; i++) inst.conflictAdj[i] = new HashSet<>();
        for (int[] e : inst.conflicts) {
            inst.conflictAdj[e[0]].add(e[1]);
            inst.conflictAdj[e[1]].add(e[0]);
        }

        return inst;
    }
}
