/**
 * Implements the extended penalty function P(σ).
 *
 * DESIGN RATIONALE (Task 2):
 *
 * P(σ) = P_base(σ) + λ₁·P_imbalance(σ) + λ₂·P_sla_risk(σ) + λ₃·P_gpu_frag(σ)
 *
 * P_base(σ)       = Σᵢ w(tᵢ) · (σ(tᵢ) + 1)
 *   Classic weighted delay: high-priority tasks in early slots pay less.
 *   We use 1-indexed slot (σ+1) so that slot-0 has nonzero cost; a task in
 *   slot 0 still has delay cost equal to its weight, preserving ordering.
 *
 * P_imbalance(σ)  = Σ_s  max_d [ usage(s,d) / C(s,d) ] − min_d [ usage(s,d) / C(s,d) ]
 *   Penalises slots where one resource is nearly exhausted while another
 *   is idle — an operational concern in shared GPU clusters (one hot
 *   dimension stalls the queue while other resources sit unused).
 *   Minimising this encourages balanced bin packing.
 *
 * P_sla_risk(σ)   = Σᵢ w(tᵢ) · [ (σ(tᵢ) − lᵢ) / (uᵢ − lᵢ + 1) ]
 *   Measures how close each assignment is to its SLA upper boundary,
 *   normalised to [0,1]. A task assigned at its latest allowed slot
 *   contributes its full weight; a task at its earliest slot contributes 0.
 *   Captures the risk that a re-run (due to upstream failure) would miss
 *   SLA — a real concern in bureau API gateway pipelines where retries are
 *   common.
 *
 * P_gpu_frag(σ)   = Σ_s  [ GPU_capacity(s) − GPU_used(s) ] · has_gpu_task(s)
 *   Penalises slots that host at least one GPU task but leave significant
 *   GPU capacity unused.  GPU accelerators are the most expensive resource
 *   in a credit ML pipeline (OCR, Fraud Score); partial allocation wastes
 *   expensive hardware.  has_gpu_task(s) = 1 if any task in slot s uses GPU > 0.
 *
 * Weights: λ₁ = 0.5, λ₂ = 1.0, λ₃ = 0.2 (tuned to keep penalty comparable
 * to P_base; can be made configurable).
 */
public class PenaltyCalculator {

    private static final double LAMBDA_IMBALANCE = 0.5;
    private static final double LAMBDA_SLA_RISK  = 1.0;
    private static final double LAMBDA_GPU_FRAG  = 0.2;

    private final InstanceData inst;

    public PenaltyCalculator(InstanceData inst) {
        this.inst = inst;
    }

    /**
     * Compute the full penalty P(σ) for a complete assignment.
     *
     * @param assignment assignment[i] = slot index (0-based)
     * @return total penalty
     */
    public double compute(int[] assignment) {
        double base       = computeBase(assignment);
        double imbalance  = computeImbalance(assignment);
        double slaRisk    = computeSlaRisk(assignment);
        double gpuFrag    = computeGpuFrag(assignment);
        return base
             + LAMBDA_IMBALANCE * imbalance
             + LAMBDA_SLA_RISK  * slaRisk
             + LAMBDA_GPU_FRAG  * gpuFrag;
    }

    /**
     * P_base = Σᵢ w(tᵢ) · (σ(tᵢ) + 1)
     * 1-indexed so that no assignment has zero cost.
     */
    public double computeBase(int[] assignment) {
        double sum = 0;
        for (int i = 0; i < inst.n; i++) {
            if (assignment[i] >= 0) {
                sum += inst.weights[i] * (assignment[i] + 1);
            }
        }
        return sum;
    }

    /**
     * P_imbalance = Σ_s  max_d(utilisation(s,d)) − min_d(utilisation(s,d))
     * Only computed over dimensions with nonzero capacity.
     */
    public double computeImbalance(int[] assignment) {
        double sum = 0;
        // Aggregate resource usage per slot
        double[][] usage = new double[inst.K][inst.d];
        for (int i = 0; i < inst.n; i++) {
            int s = assignment[i];
            if (s >= 0) {
                for (int dd = 0; dd < inst.d; dd++) {
                    usage[s][dd] += inst.resources[i][dd];
                }
            }
        }
        for (int s = 0; s < inst.K; s++) {
            double maxUtil = 0, minUtil = Double.MAX_VALUE;
            for (int dd = 0; dd < inst.d; dd++) {
                double cap = inst.capacities[s][dd];
                if (cap > 0) {
                    double util = usage[s][dd] / cap;
                    maxUtil = Math.max(maxUtil, util);
                    minUtil = Math.min(minUtil, util);
                }
            }
            if (minUtil == Double.MAX_VALUE) minUtil = 0;
            sum += (maxUtil - minUtil);
        }
        return sum;
    }

    /**
     * P_sla_risk = Σᵢ w(tᵢ) · (σ(tᵢ) − lᵢ) / (uᵢ − lᵢ + 1)
     */
    public double computeSlaRisk(int[] assignment) {
        double sum = 0;
        for (int i = 0; i < inst.n; i++) {
            int s = assignment[i];
            if (s >= 0) {
                int lo = inst.windows[i][0];
                int hi = inst.windows[i][1];
                int window = hi - lo + 1;
                double risk = (double)(s - lo) / window;
                sum += inst.weights[i] * risk;
            }
        }
        return sum;
    }

    /**
     * P_gpu_frag = Σ_s  (GPU_cap(s) − GPU_used(s)) · 1[any GPU task in s]
     */
    public double computeGpuFrag(int[] assignment) {
        int GPU_DIM = 2; // index of GPU in the resource vector
        double[] gpuUsed = new double[inst.K];
        boolean[] hasGpuTask = new boolean[inst.K];
        for (int i = 0; i < inst.n; i++) {
            int s = assignment[i];
            if (s >= 0) {
                gpuUsed[s] += inst.resources[i][GPU_DIM];
                if (inst.resources[i][GPU_DIM] > 0) {
                    hasGpuTask[s] = true;
                }
            }
        }
        double sum = 0;
        for (int s = 0; s < inst.K; s++) {
            if (hasGpuTask[s]) {
                sum += (inst.capacities[s][GPU_DIM] - gpuUsed[s]);
            }
        }
        return sum;
    }
}
