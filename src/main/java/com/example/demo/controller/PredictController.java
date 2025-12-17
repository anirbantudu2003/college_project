package com.example.demo.controller;

import com.example.demo.model.InsuranceRequest;
import com.example.demo.model.InsuranceResponse;
import com.example.demo.service.MLPredictionService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class PredictController {

    private final MLPredictionService mlPredictionService;

    public PredictController(MLPredictionService mlPredictionService) {
        this.mlPredictionService = mlPredictionService;
    }

    @PostMapping(value = "/predict", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public InsuranceResponse predict(@RequestBody InsuranceRequest req) {
        // Validate required fields
        if (req.getAge() == null || req.getAge() <= 0) {
            return new InsuranceResponse("Error: Age must be greater than 0");
        }
        if (req.getBmi() == null || req.getBmi() <= 0) {
            return new InsuranceResponse("Error: BMI must be greater than 0");
        }
        if (req.getGender() == null || req.getGender().trim().isEmpty()) {
            return new InsuranceResponse("Error: Gender is required");
        }
        if (req.getLocation() == null || req.getLocation().trim().isEmpty()) {
            return new InsuranceResponse("Error: Location is required");
        }

        // Get ML model prediction with selected model
        String selectedModel = req.getModel();
        if (selectedModel == null || selectedModel.trim().isEmpty()) {
            selectedModel = "random_forest"; // Default model
        }
        
        Map<String, Object> predictionResult = mlPredictionService.predictInsuranceCost(
                req.getAge(),
                req.getGender(),
                req.getBmi(),
                req.getKids() != null ? req.getKids() : 0,
                req.getSmoker() != null ? req.getSmoker() : false,
                req.getLocation(),
                selectedModel
        );
        
        String usedModel = (String) predictionResult.get("model");
        double predictedCost = ((Number) predictionResult.get("prediction")).doubleValue();

        // Generate a friendly response with suggestions
        StringBuilder response = new StringBuilder();
        response.append("=== Insurance Cost Estimate ===\n");
        response.append(String.format("Model Used: %s\n", usedModel.replace("_", " ").toUpperCase()));
        response.append(String.format("Estimated Annual Cost: $%.2f\n\n", predictedCost));
        
        response.append("=== Personalized Suggestions to Reduce Cost ===\n");
        
        // Generate suggestions based on input
        if (req.getSmoker() != null && req.getSmoker()) {
            response.append("1. Consider quitting smoking - this could reduce your premium by up to 50%\n");
        } else {
            response.append("1. Maintain your non-smoker status - this keeps your premiums lower\n");
        }

        if (req.getBmi() != null && req.getBmi() > 30) {
            response.append("2. Work on reducing your BMI through diet and exercise - lower BMI means lower costs\n");
        } else if (req.getBmi() != null && req.getBmi() > 25) {
            response.append("2. Maintain a healthy BMI to keep insurance costs stable\n");
        } else {
            response.append("2. Keep maintaining your healthy BMI - this is excellent for your health and insurance costs\n");
        }

        response.append("3. Schedule regular health check-ups to prevent chronic conditions\n");

        if (req.getExistingCondition() != null && !req.getExistingCondition().trim().isEmpty()) {
            response.append("\nNote: Your existing condition(s) may impact the final premium. Please consult with an insurance agent for detailed information.\n");
        }

        return new InsuranceResponse(response.toString());
    }
}
