import javax.swing.*;
import java.awt.*;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.SerializationHelper;
import java.util.ArrayList;

public class TemperatureDisplay extends JFrame {
    private static final int DAYS_TO_SHOW = 3;
    private JPanel mainPanel;
    private Classifier minTempModel;
    private Classifier maxTempModel;
    private Instances minDataset;
    private Instances maxDataset;
    private double currentMinTemp = 15.0;  // Starting with example temperature
    private double currentMaxTemp = 25.0;

    public TemperatureDisplay() {
        setTitle("Temperature Forecast");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setMinimumSize(new Dimension(400, 600));
        setPreferredSize(new Dimension(500, 700));
        getContentPane().setBackground(new Color(240, 248, 255));  // Light blue background

        // Initialize the main panel with vertical layout
        mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.setBackground(new Color(240, 248, 255));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        // Create a scroll pane for the main panel
        JScrollPane scrollPane = new JScrollPane(mainPanel);
        scrollPane.setBorder(null);
        scrollPane.setBackground(new Color(240, 248, 255));
        scrollPane.getVerticalScrollBar().setUnitIncrement(16);
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);

        // Add title
        JLabel titleLabel = new JLabel("Temperature Forecast");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 28));
        titleLabel.setForeground(new Color(30, 144, 255));
        titleLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
        mainPanel.add(titleLabel);
        mainPanel.add(Box.createVerticalStrut(20));

        try {
            loadModels();
            setupDatasets();
            updateDisplay();
        } catch (Exception e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error loading models: " + e.getMessage());
        }

        add(scrollPane);
    }

    private void loadModels() throws Exception {
        // Load the best performing models (you can modify this to load specific models)
        minTempModel = (Classifier) SerializationHelper.read("model/min/random_forest.model");
        maxTempModel = (Classifier) SerializationHelper.read("model/max/random_forest.model");
    }

    private void setupDatasets() {
        // Create attributes for min and max temperatures
        ArrayList<Attribute> minAttributes = new ArrayList<>();
        minAttributes.add(new Attribute("current_min_temp"));
        minAttributes.add(new Attribute("next_min_temp"));
        minDataset = new Instances("MinTemperatureData", minAttributes, 0);
        minDataset.setClassIndex(1);

        ArrayList<Attribute> maxAttributes = new ArrayList<>();
        maxAttributes.add(new Attribute("current_max_temp"));
        maxAttributes.add(new Attribute("next_max_temp"));
        maxDataset = new Instances("MaxTemperatureData", maxAttributes, 0);
        maxDataset.setClassIndex(1);
    }

    private void updateDisplay() {
        mainPanel.removeAll();
        mainPanel.add(Box.createVerticalStrut(20));  // Space after title

        SimpleDateFormat dateFormat = new SimpleDateFormat("EEEE, MMM dd");
        Calendar calendar = Calendar.getInstance();
        calendar.add(Calendar.DAY_OF_MONTH, -DAYS_TO_SHOW);

        try {
            double minTemp = currentMinTemp;
            double maxTemp = currentMaxTemp;
            
            for (int i = 0; i < 7; i++) {
                String date = dateFormat.format(calendar.getTime());
                boolean isToday = (i == DAYS_TO_SHOW);
                
                JPanel dayPanel = createDayPanel(date, getRelativeDay(i), minTemp, maxTemp, isToday);
                mainPanel.add(dayPanel);
                mainPanel.add(Box.createVerticalStrut(15));  // Space between days
                
                // Predict next day's temperatures
                minTemp = predictTemperature(minDataset, minTempModel, minTemp);
                maxTemp = predictTemperature(maxDataset, maxTempModel, maxTemp);
                
                calendar.add(Calendar.DAY_OF_MONTH, 1);
            }
        } catch (Exception e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error making predictions: " + e.getMessage());
        }

        mainPanel.revalidate();
        mainPanel.repaint();
    }

    private double predictTemperature(Instances dataset, Classifier model, double currentTemp) throws Exception {
        DenseInstance instance = new DenseInstance(1.0, new double[]{currentTemp, Double.NaN});
        instance.setDataset(dataset);
        return model.classifyInstance(instance);
    }

    private String getRelativeDay(int dayOffset) {
        switch (dayOffset) {
            case 0: return "3 days ago";
            case 1: return "2 days ago";
            case 2: return "Yesterday";
            case 3: return "Today";
            case 4: return "Tomorrow";
            case 5: return "In 2 days";
            case 6: return "In 3 days";
            default: return "";
        }
    }

    private JPanel createDayPanel(String date, String relativeDay, double minTemp, double maxTemp, boolean isToday) {
        JPanel dayPanel = new JPanel();
        dayPanel.setLayout(new BoxLayout(dayPanel, BoxLayout.Y_AXIS));
        dayPanel.setBackground(isToday ? new Color(230, 240, 255) : new Color(240, 248, 255));
        dayPanel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(isToday ? new Color(30, 144, 255) : Color.LIGHT_GRAY, 2),
            BorderFactory.createEmptyBorder(15, 20, 15, 20)
        ));
        dayPanel.setMaximumSize(new Dimension(450, 120));
        dayPanel.setAlignmentX(Component.CENTER_ALIGNMENT);

        // Date label
        JLabel dateLabel = new JLabel(date);
        dateLabel.setFont(new Font("Arial", Font.BOLD, 16));
        dateLabel.setForeground(isToday ? new Color(30, 144, 255) : Color.BLACK);
        dateLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        // Relative day label
        JLabel relativeDayLabel = new JLabel("(" + relativeDay + ")");
        relativeDayLabel.setFont(new Font("Arial", isToday ? Font.BOLD : Font.PLAIN, 14));
        relativeDayLabel.setForeground(isToday ? new Color(30, 144, 255) : Color.DARK_GRAY);
        relativeDayLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        // Temperature labels
        JLabel minLabel = new JLabel(String.format("Min: %.1f°C", minTemp));
        minLabel.setFont(new Font("Arial", Font.PLAIN, 15));
        minLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        JLabel maxLabel = new JLabel(String.format("Max: %.1f°C", maxTemp));
        maxLabel.setFont(new Font("Arial", Font.PLAIN, 15));
        maxLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        dayPanel.add(dateLabel);
        dayPanel.add(Box.createVerticalStrut(5));
        dayPanel.add(relativeDayLabel);
        dayPanel.add(Box.createVerticalStrut(10));
        dayPanel.add(minLabel);
        dayPanel.add(Box.createVerticalStrut(5));
        dayPanel.add(maxLabel);

        return dayPanel;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            TemperatureDisplay display = new TemperatureDisplay();
            display.setVisible(true);
        });
    }
}
