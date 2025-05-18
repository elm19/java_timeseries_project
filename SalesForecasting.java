import weka.classifiers.timeseries.WekaForecaster;
import weka.classifiers.evaluation.NumericPrediction;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.List;
import weka.core.Version;

public class SalesForecasting {
    public static void main(String[] args) throws Exception {
        System.out.println("Weka Version: " + Version.VERSION);
        // Load the time series data (assume data/sales.arff exists)
        DataSource source = new DataSource("data/sales.arff");
        Instances data = source.getDataSet();

        // Create the forecaster and set the target field
        WekaForecaster forecaster = new WekaForecaster();
        forecaster.setFieldsToForecast("sales");

        // Build the model
        forecaster.buildForecaster(data, System.out);
        forecaster.primeForecaster(data);

        // Forecast for 5 future steps
        List<List<NumericPrediction>> forecast = forecaster.forecast(5, System.out);
        System.out.println("Forecasted values:");
        for (List<NumericPrediction> preds : forecast) {
            for (NumericPrediction pred : preds) {
                System.out.println(pred.predicted());
            }
        }
    }
}
