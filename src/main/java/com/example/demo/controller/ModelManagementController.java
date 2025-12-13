package com.example.demo.controller;

import com.example.demo.service.ModelRetrainingService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

/**
 * REST API endpoints for dynamic model retraining and monitoring
 */
@RestController
@RequestMapping("/api/model")
public class ModelManagementController {

    private final ModelRetrainingService retrainingService;

    public ModelManagementController(ModelRetrainingService retrainingService) {
        this.retrainingService = retrainingService;
    }

    /**
     * GET /api/model/metrics - Get current model performance metrics
     */
    @GetMapping("/metrics")
    public ResponseEntity<Map<String, Object>> getMetrics() {
        Map<String, Object> metrics = retrainingService.getModelMetrics();
        return ResponseEntity.ok(metrics);
    }

    /**
     * GET /api/model/datasets - Get available Kaggle datasets for training
     */
    @GetMapping("/datasets")
    public ResponseEntity<Map<String, Object>> getDatasets() {
        Map<String, Object> datasets = retrainingService.getAvailableDatasets();
        return ResponseEntity.ok(datasets);
    }

    /**
     * POST /api/model/retrain - Trigger model retraining
     * 
     * Request body:
     * {
     *   "source": "synthetic|kaggle",
     *   "datasetId": "mirichoi/insurance",  (required if source=kaggle)
     *   "nSamples": 500,                    (for synthetic data)
     *   "nEstimators": 100,                 (Random Forest hyperparameter)
     *   "maxDepth": 10                      (Random Forest hyperparameter)
     * }
     * 
     * Example:
     * {
     *   "source": "synthetic",
     *   "nSamples": 1000,
     *   "nEstimators": 150,
     *   "maxDepth": 12
     * }
     * 
     * Or:
     * {
     *   "source": "kaggle",
     *   "datasetId": "mirichoi/insurance"
     * }
     */
    @PostMapping("/retrain")
    public ResponseEntity<Map<String, Object>> retrainModel(@RequestBody RetrainingRequest request) {
        if (request.getSource() == null || request.getSource().trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of(
                "status", "error",
                "message", "source is required (synthetic or kaggle)"
            ));
        }

        if ("kaggle".equals(request.getSource()) && 
            (request.getDatasetId() == null || request.getDatasetId().trim().isEmpty())) {
            return ResponseEntity.badRequest().body(Map.of(
                "status", "error",
                "message", "datasetId is required when source=kaggle"
            ));
        }

        Map<String, Object> result = retrainingService.retrainModel(
            request.getSource(),
            request.getDatasetId(),
            request.getNSamples(),
            request.getNEstimators(),
            request.getMaxDepth()
        );

        return ResponseEntity.ok(result);
    }

    /**
     * Request body DTO for retraining
     */
    public static class RetrainingRequest {
        private String source;          // "synthetic" or "kaggle"
        private String datasetId;       // Kaggle dataset ID
        private Integer nSamples;       // Number of samples (synthetic only)
        private Integer nEstimators;    // Random Forest hyperparameter
        private Integer maxDepth;       // Random Forest hyperparameter

        // Getters
        public String getSource() { return source; }
        public String getDatasetId() { return datasetId; }
        public Integer getNSamples() { return nSamples; }
        public Integer getNEstimators() { return nEstimators; }
        public Integer getMaxDepth() { return maxDepth; }

        // Setters
        public void setSource(String source) { this.source = source; }
        public void setDatasetId(String datasetId) { this.datasetId = datasetId; }
        public void setNSamples(Integer nSamples) { this.nSamples = nSamples; }
        public void setNEstimators(Integer nEstimators) { this.nEstimators = nEstimators; }
        public void setMaxDepth(Integer maxDepth) { this.maxDepth = maxDepth; }
    }
}
