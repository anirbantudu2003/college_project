package com.example.demo.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Service for dynamic model retraining using different data sources and hyperparameters.
 */
@Service
public class ModelRetrainingService {

    private static final String TRAINING_SCRIPT = "train_model_dynamic.py";
    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Retrain the model with specified parameters
     */
    public Map<String, Object> retrainModel(String source, String datasetId, Integer nSamples, 
                                           Integer nEstimators, Integer maxDepth) {
        try {
            // Build Python command
            List<String> command = new ArrayList<>();
            command.add(getPythonExecutable());
            command.add(TRAINING_SCRIPT);
            command.add("--source");
            command.add(source != null ? source : "synthetic");
            
            if ("kaggle".equals(source) && datasetId != null) {
                command.add("--dataset");
                command.add(datasetId);
            }
            
            if (nSamples != null && nSamples > 0) {
                command.add("--n_samples");
                command.add(String.valueOf(nSamples));
            }
            
            if (nEstimators != null && nEstimators > 0) {
                command.add("--n_estimators");
                command.add(String.valueOf(nEstimators));
            }
            
            if (maxDepth != null && maxDepth > 0) {
                command.add("--max_depth");
                command.add(String.valueOf(maxDepth));
            }
            
            System.out.println("[*] Starting model retraining with command: " + String.join(" ", command));
            
            // Execute training script
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();
            
            // Capture output
            StringBuilder output = new StringBuilder();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                System.out.println(line);
            }
            
            int exitCode = process.waitFor();
            
            if (exitCode != 0) {
                return Map.of(
                    "status", "error",
                    "message", "Training script failed with exit code " + exitCode,
                    "output", output.toString()
                );
            }
            
            // Parse result from output
            String resultStr = output.toString();
            
            // Try to extract JSON result
            int jsonStart = resultStr.lastIndexOf('{');
            int jsonEnd = resultStr.lastIndexOf('}') + 1;
            
            if (jsonStart >= 0 && jsonEnd > jsonStart) {
                String jsonPart = resultStr.substring(jsonStart, jsonEnd);
                try {
                    Map<String, Object> result = objectMapper.readValue(jsonPart, Map.class);
                    return result;
                } catch (Exception e) {
                    System.err.println("Failed to parse training result JSON: " + e.getMessage());
                }
            }
            
            return Map.of(
                "status", "success",
                "message", "Model trained successfully",
                "output", resultStr
            );
            
        } catch (Exception e) {
            System.err.println("Error during model retraining: " + e.getMessage());
            e.printStackTrace();
            return Map.of(
                "status", "error",
                "message", "Retraining failed: " + e.getMessage()
            );
        }
    }

    /**
     * Get current model metrics from config file
     */
    public Map<String, Object> getModelMetrics() {
        try {
            String configPath = "src/main/resources/models/model_config.json";
            if (Files.exists(Paths.get(configPath))) {
                String content = new String(Files.readAllBytes(Paths.get(configPath)));
                Map<String, Object> config = objectMapper.readValue(content, Map.class);
                return Map.of(
                    "status", "success",
                    "metrics", config.get("metrics"),
                    "features", config.get("feature_names"),
                    "encoders", config.get("encoders")
                );
            } else {
                return Map.of(
                    "status", "warning",
                    "message", "No model configuration found. Train a model first."
                );
            }
        } catch (Exception e) {
            return Map.of(
                "status", "error",
                "message", "Failed to read model metrics: " + e.getMessage()
            );
        }
    }

    /**
     * Get available Kaggle datasets for training
     */
    public Map<String, Object> getAvailableDatasets() {
        return Map.of(
            "status", "success",
            "datasets", List.of(
                Map.of(
                    "id", "mirichoi/insurance",
                    "name", "Medical Insurance Dataset",
                    "description", "Insurance charges by age, gender, BMI, smoking, region",
                    "samples", 1338
                ),
                Map.of(
                    "id", "easonlai/health_insurance_cost_prediction",
                    "name", "Health Insurance Cost Prediction",
                    "description", "Health insurance cost prediction dataset",
                    "samples", 1338
                ),
                Map.of(
                    "id", "noordeen/insurance-premium-prediction",
                    "name", "Insurance Premium Prediction",
                    "description", "Insurance premium data",
                    "samples", 4000
                )
            ),
            "note", "Datasets require Kaggle API credentials to download"
        );
    }

    /**
     * Get the Python executable path
     */
    private String getPythonExecutable() {
        String[] pythonPaths = {
                "C:/Users/ASIF EBRAHIM/Downloads/demo/.venv/Scripts/python.exe",
                ".venv/Scripts/python.exe",
                "python",
                "python3"
        };

        for (String path : pythonPaths) {
            try {
                ProcessBuilder pb = new ProcessBuilder(path, "--version");
                pb.redirectErrorStream(true);
                Process p = pb.start();
                int code = p.waitFor();
                if (code == 0) {
                    return path;
                }
            } catch (Exception ignored) {}
        }

        return "python";
    }
}
