import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.awt.Color;
import java.awt.Font;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

public class TemperaturePlotter {
    private static final String DATA_FILE = "data/daily_temp.csv";
    private static final String OUTPUT_DIR = "plots/";

    public static void main(String[] args) {
        try {
            // Create output directory
            new File(OUTPUT_DIR).mkdirs();

            // Load models
            Classifier minModel = (Classifier) SerializationHelper.read("model/min/random_forest.model");
            Classifier maxModel = (Classifier) SerializationHelper.read("model/max/random_forest.model");

            // Create datasets for plotting
            XYSeries actualMinTemp = new XYSeries("Actual Min Temperature");
            XYSeries predictedMinTemp = new XYSeries("Predicted Min Temperature");
            XYSeries actualMaxTemp = new XYSeries("Actual Max Temperature");
            XYSeries predictedMaxTemp = new XYSeries("Predicted Max Temperature");

            // Read data and make predictions
            BufferedReader reader = new BufferedReader(new FileReader(DATA_FILE));
            reader.readLine(); // Skip header

            String line;
            int dayIndex = 0;
            double prevMinTemp = 0;
            double prevMaxTemp = 0;
            boolean isFirst = true;

            // Store all lines in memory first
            ArrayList<String> allLines = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    allLines.add(line);
                }
            }
            reader.close();

            // Process only the last 30 days
            int startIndex = Math.max(0, allLines.size() - 30);
            for (int i = startIndex; i < allLines.size(); i++) {
                line = allLines.get(i);
                String[] values = line.split(",");
                
                // Parse date and temperatures
                java.time.LocalDate date = java.time.LocalDate.parse(values[0]);
                double minTemp = Double.parseDouble(values[1]);
                double maxTemp = Double.parseDouble(values[2]);

                actualMinTemp.add(dayIndex, minTemp);
                actualMaxTemp.add(dayIndex, maxTemp);

                if (!isFirst) {
                    // Create instances for prediction
                    double predictedMin = makePrediction(minModel, prevMinTemp);
                    double predictedMax = makePrediction(maxModel, prevMaxTemp);
                    
                    predictedMinTemp.add(dayIndex, predictedMin);
                    predictedMaxTemp.add(dayIndex, predictedMax);
                }

                prevMinTemp = minTemp;
                prevMaxTemp = maxTemp;
                isFirst = false;
                dayIndex++;
            }

            // Create and save minimum temperature plot
            createAndSavePlot(
                actualMinTemp, 
                predictedMinTemp, 
                "Last 30 Days - Minimum Temperature: Actual vs Predicted",
                "Days",
                "Temperature (°C)",
                "min_temp_comparison.png"
            );

            // Create and save maximum temperature plot
            createAndSavePlot(
                actualMaxTemp, 
                predictedMaxTemp, 
                "Last 30 Days - Maximum Temperature: Actual vs Predicted",
                "Days",
                "Temperature (°C)",
                "max_temp_comparison.png"
            );

            System.out.println("Plots have been created successfully in the " + OUTPUT_DIR + " directory");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static double makePrediction(Classifier model, double prevTemp) throws Exception {
        // Create attributes matching the training data structure
        ArrayList<weka.core.Attribute> attributes = new ArrayList<>();
        attributes.add(new weka.core.Attribute("prev_temp"));
        attributes.add(new weka.core.Attribute("month"));
        attributes.add(new weka.core.Attribute("day"));
        attributes.add(new weka.core.Attribute("next_temp"));

        Instances dataset = new Instances("PredictionData", attributes, 0);
        dataset.setClassIndex(3); // next_temp is the target
        
        // Create instance with appropriate features
        double[] values = new double[4];
        values[0] = prevTemp;
        // Use current month and day as features (this could be improved by using actual dates)
        values[1] = java.time.LocalDate.now().getMonthValue();
        values[2] = java.time.LocalDate.now().getDayOfMonth();
        values[3] = Double.NaN; // This is what we're predicting
        
        DenseInstance instance = new DenseInstance(1.0, values);
        instance.setDataset(dataset);
        
        return model.classifyInstance(instance);
    }

    private static void createAndSavePlot(
            XYSeries actualSeries, 
            XYSeries predictedSeries, 
            String title,
            String xLabel,
            String yLabel,
            String filename) throws Exception {
        
        // Only keep the last 30 days of data
        int dataSize = actualSeries.getItemCount();
        if (dataSize > 30) {
            for (int i = 0; i < dataSize - 30; i++) {
                actualSeries.remove(0);
                if (predictedSeries.getItemCount() > 0) {
                    predictedSeries.remove(0);
                }
            }
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(actualSeries);
        dataset.addSeries(predictedSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            xLabel,
            yLabel,
            dataset
        );

        // Customize the plot
        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        
        // Set series colors and styles
        renderer.setSeriesPaint(0, new Color(0, 102, 204));    // Actual temperature - dark blue
        renderer.setSeriesPaint(1, new Color(255, 102, 102));  // Predicted temperature - coral red
        renderer.setSeriesStroke(0, new java.awt.BasicStroke(2.5f));
        renderer.setSeriesStroke(1, new java.awt.BasicStroke(2.5f, java.awt.BasicStroke.CAP_ROUND, java.awt.BasicStroke.JOIN_ROUND, 
                                                          1.0f, new float[]{6.0f, 6.0f}, 0.0f));  // Dashed line for predictions

        // Show shapes at data points
        renderer.setSeriesShapesVisible(0, true);
        renderer.setSeriesShapesVisible(1, true);
        renderer.setSeriesShape(0, new java.awt.geom.Ellipse2D.Double(-3, -3, 6, 6));
        renderer.setSeriesShape(1, new java.awt.geom.Rectangle2D.Double(-3, -3, 6, 6));

        // Enable grid lines
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinesVisible(true);
        plot.setRangeGridlinesVisible(true);
        
        // Set background
        plot.setBackgroundPaint(Color.WHITE);
        chart.setBackgroundPaint(Color.WHITE);
        
        // Customize the range axis (temperature)
        plot.getRangeAxis().setTickLabelFont(new Font("Arial", Font.PLAIN, 12));
        plot.getRangeAxis().setLabelFont(new Font("Arial", Font.BOLD, 14));
        
        // Customize the domain axis (days)
        plot.getDomainAxis().setTickLabelFont(new Font("Arial", Font.PLAIN, 12));
        plot.getDomainAxis().setLabelFont(new Font("Arial", Font.BOLD, 14));

        plot.setRenderer(renderer);

        // Add legend with custom font
        chart.getLegend().setItemFont(new Font("Arial", Font.PLAIN, 12));
        chart.getTitle().setFont(new Font("Arial", Font.BOLD, 16));

        // Save the chart as PNG with higher resolution
        ChartUtils.saveChartAsPNG(
            new File(OUTPUT_DIR + filename),
            chart,
            1000,   // width
            600     // height
        );
    }
}