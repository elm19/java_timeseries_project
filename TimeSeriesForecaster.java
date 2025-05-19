import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;
import java.io.File;

public class TimeSeriesForecaster {
    // Configuration constants
    private static final String DATA_FILE = "data/daily_temp.csv";
    private static final double EXAMPLE_MIN_TEMP = 15.0;
    private static final double EXAMPLE_MAX_TEMP = 25.0;

    private static class ModelMetrics {
        double rmse;
        double mae;
        double correlation;
        weka.classifiers.Classifier model;

        ModelMetrics(double rmse, double mae, double correlation, weka.classifiers.Classifier model) {
            this.rmse = rmse;
            this.mae = mae;
            this.correlation = correlation;
            this.model = model;
        }
    }

    private static ModelMetrics evaluateModel(String modelName, weka.classifiers.Classifier model, Instances dataset) throws Exception {
        // Perform 10-fold cross-validation
        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(model, dataset, 10, new Random(1));

        // Print evaluation metrics
        System.out.println("\n=== " + modelName + " ===");
        System.out.println("Root Mean Squared Error: " + eval.rootMeanSquaredError());
        System.out.println("Mean Absolute Error: " + eval.meanAbsoluteError());
        System.out.println("Correlation Coefficient: " + eval.correlationCoefficient());
        System.out.println("Relative Absolute Error: " + eval.relativeAbsoluteError() + "%");

        return new ModelMetrics(eval.rootMeanSquaredError(), 
                              eval.meanAbsoluteError(), 
                              eval.correlationCoefficient(),
                              model);
    }

    private static String findBestModel(Map<String, ModelMetrics> models) {
        String bestModel = null;
        double bestScore = Double.POSITIVE_INFINITY;

        for (Map.Entry<String, ModelMetrics> entry : models.entrySet()) {
            // Combined score (lower is better) - weighted average of normalized metrics
            double score = entry.getValue().rmse * 0.4 + 
                          entry.getValue().mae * 0.4 + 
                          (1 - entry.getValue().correlation) * 0.2;
            
            if (score < bestScore) {
                bestScore = score;
                bestModel = entry.getKey();
            }
        }

        return bestModel;
    }

    // Helper method to create dataset attributes
    private static ArrayList<Attribute> createAttributes(String prefix) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("current_" + prefix + "_temp"));
        attributes.add(new Attribute("next_" + prefix + "_temp"));
        return attributes;
    }

    // Helper method to load data from CSV
    private static void loadDataFromCSV(BufferedReader reader, Instances minDataset, Instances maxDataset) throws Exception {
        String line;
        reader.readLine(); // Skip header
        
        while ((line = reader.readLine()) != null) {
            String[] values = line.split(",");
            if (values.length < 3) {
                System.err.println("Skipping malformed line: " + line);
                continue;
            }

            double minTemp = Double.parseDouble(values[1]);
            double maxTemp = Double.parseDouble(values[2]);

            // Add to datasets
            addToDataset(minDataset, minTemp);
            addToDataset(maxDataset, maxTemp);
        }
    }

    // Helper method to add temperature to dataset
    private static void addToDataset(Instances dataset, double temp) {
        dataset.add(new DenseInstance(1.0, new double[]{temp, Double.NaN}));
        if (dataset.size() > 1) {
            dataset.instance(dataset.size() - 2).setValue(1, temp);
        }
    }

    // Helper method to train and evaluate a specific type of model
    private static void trainAndEvaluateModel(String modelType, weka.classifiers.Classifier model, 
                                            Instances dataset, Map<String, ModelMetrics> models,
                                            String temperatureType) throws Exception {
        model.buildClassifier(dataset);
        models.put(modelType, evaluateModel(modelType, model, dataset));
        
        // Create directory if it doesn't exist
        String modelDir = "model/" + temperatureType.toLowerCase();
        new File(modelDir).mkdirs();
        
        // Save the trained model
        String modelPath = modelDir + "/" + modelType.toLowerCase().replace(" ", "_") + ".model";
        SerializationHelper.write(modelPath, model);
        System.out.println("Saved " + temperatureType + " " + modelType + " model to: " + modelPath);
    }

    // Helper method to make predictions
    private static double predict(Instances dataset, Map<String, ModelMetrics> models, 
                                String bestModel, double currentTemp) throws Exception {
        DenseInstance instance = new DenseInstance(1.0, new double[]{currentTemp, Double.NaN});
        instance.setDataset(dataset);
        return models.get(bestModel).model.classifyInstance(instance);
    }

    public static void main(String[] args) {
        try {
            // Initialize datasets
            Instances minTempDataset = new Instances("MinTemperatureData", createAttributes("min"), 0);
            Instances maxTempDataset = new Instances("MaxTemperatureData", createAttributes("max"), 0);
            minTempDataset.setClassIndex(1);
            maxTempDataset.setClassIndex(1);

            // Load data
            try (BufferedReader reader = new BufferedReader(new FileReader(DATA_FILE))) {
                loadDataFromCSV(reader, minTempDataset, maxTempDataset);
            }

            // Remove last incomplete instances
            minTempDataset.delete(minTempDataset.size() - 1);
            maxTempDataset.delete(maxTempDataset.size() - 1);

            // Train and evaluate models for minimum temperature
            System.out.println("\n=== Minimum Temperature Models ===");
            Map<String, ModelMetrics> minTempModels = new HashMap<>();
            trainAndEvaluateModel("Linear Regression", new LinearRegression(), minTempDataset, minTempModels, "min");
            trainAndEvaluateModel("Random Forest", new RandomForest(), minTempDataset, minTempModels, "min");
            trainAndEvaluateModel("Support Vector Regression", new SMOreg(), minTempDataset, minTempModels, "min");

            // Train and evaluate models for maximum temperature
            System.out.println("\n=== Maximum Temperature Models ===");
            Map<String, ModelMetrics> maxTempModels = new HashMap<>();
            trainAndEvaluateModel("Linear Regression", new LinearRegression(), maxTempDataset, maxTempModels, "max");
            trainAndEvaluateModel("Random Forest", new RandomForest(), maxTempDataset, maxTempModels, "max");
            trainAndEvaluateModel("Support Vector Regression", new SMOreg(), maxTempDataset, maxTempModels, "max");

            // Find best models
            String bestMinTempModel = findBestModel(minTempModels);
            String bestMaxTempModel = findBestModel(maxTempModels);

            // Print results
            System.out.println("\n=== Model Comparison Results ===");
            printModelResults("minimum", bestMinTempModel, minTempModels.get(bestMinTempModel));
            printModelResults("maximum", bestMaxTempModel, maxTempModels.get(bestMaxTempModel));

            // Make predictions
            System.out.println("\n=== Predictions for next day ===");
            System.out.println("Current min temp: " + EXAMPLE_MIN_TEMP);
            System.out.println("Current max temp: " + EXAMPLE_MAX_TEMP);

            double predictedMinTemp = predict(minTempDataset, minTempModels, bestMinTempModel, EXAMPLE_MIN_TEMP);
            double predictedMaxTemp = predict(maxTempDataset, maxTempModels, bestMaxTempModel, EXAMPLE_MAX_TEMP);

            System.out.println("Predicted min temperature: " + predictedMinTemp);
            System.out.println("Predicted max temperature: " + predictedMaxTemp);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Helper method to print model results
    private static void printModelResults(String type, String modelName, ModelMetrics metrics) {
        System.out.println("Best model for " + type + " temperature: " + modelName);
        System.out.println("- RMSE: " + metrics.rmse);
        System.out.println("- MAE: " + metrics.mae);
        System.out.println("- Correlation: " + metrics.correlation);
    }
}