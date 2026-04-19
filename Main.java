import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.util.*;

/**
 * Entry point for the MSME Credit Pipeline Scheduler.
 * Reads a JSON instance file and writes a JSON solution file.
 *
 * Algorithm: Priority-Weighted DSATUR with Resource-Aware Backtracking (PWDRAB)
 */
public class Main {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java -jar scheduler.jar <instance.json> [output.json]");
            System.exit(1);
        }

        String inputPath = args[0];
        String outputPath = args.length > 1 ? args[1] : "output.json";

        ObjectMapper mapper = new ObjectMapper();
        InstanceData instance = InstanceData.fromFile(inputPath, mapper);

        long startMs = System.currentTimeMillis();
        Scheduler scheduler = new Scheduler(instance);
        SchedulerResult result = scheduler.solve();
        long endMs = System.currentTimeMillis();

        result.runtimeMs = (int)(endMs - startMs);

        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputPath), result.toMap());
        System.out.println("Done. Feasible=" + result.feasible + " Penalty=" + String.format("%.4f", result.penalty)
                + " Runtime=" + result.runtimeMs + "ms");
    }
}
