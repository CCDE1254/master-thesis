/**
 * 
 */
package master.thesis.experiments;

import java.awt.Color;
import java.awt.Font;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import master.thesis.adaptive.importance.sampling.DoubleBarrierKnockOutCallAdaptiveImportanceSampling;
import master.thesis.barrieroption.DoubleBarrierKnockOutCall;
import master.thesis.gradient.descent.AdaGradAISCall;
import master.thesis.gradient.descent.AdaGradISCall;
import master.thesis.importance.sampling.DoubleBarrierKnockOutCallImportanceSampling;
import master.thesis.randomnumber.MersenneTwisterSequence;
import master.thesis.timediscretization.TimeDiscretizationWithEqualTimeStepSize;

/**
 * @author QuanLiu
 *
 */
public class EstimationErrorPlot extends JFrame {

	static double initialStockPrice = 100.0;
	static double riskFreeRate = 0.02;
	static double volatilityTerm = 0.15;
	
	static double maturity = 10.0;//set evaluation time to be 0
	static double strike = 150.0;
	
	static double upperBoundFactorB = 150.0;
	static double upperBoundExponentialDelta1 = 0.06;
	static double lowerBoundFactorA = 50.0;
	static double lowerBoundExponentialDelta2 = 0.06;
	
	static int numberOfSimulations = 1000;
	static int numberOfTimeSteps = 20;
	
	static int numberOfFirstHiddenLayerNeurons=10;
	static int numberOfSecondHiddenLayerNeurons=5;
	
	static double learningRate = 0.0005;
	static int numberOfIterationTimes = 50;
	static double epsilon = 0.000000000000001;
	
    public static void main(String[] args) {

    	
    	
        SwingUtilities.invokeLater(() -> {
        	EstimationErrorPlot ex = new EstimationErrorPlot();
            ex.setVisible(true);
        });
    }
    
    public EstimationErrorPlot() {

        initUI();
    }

    private void initUI() {

        XYDataset dataset = createDataset();
        JFreeChart chart = createChart(dataset);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setBorder(BorderFactory.createEmptyBorder(15, 15, 15, 15));
        chartPanel.setBackground(Color.white);
        add(chartPanel);

        pack();
        setTitle("Line chart");
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    private XYDataset createDataset() {

    	TimeDiscretizationWithEqualTimeStepSize time = new TimeDiscretizationWithEqualTimeStepSize(0, maturity, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		MersenneTwisterSequence randomnumber = new MersenneTwisterSequence(numberOfSimulations, timeSeries);
		double[][] randomNumberMatrix = randomnumber.getRandomNumberRealizations();
		
		//analytic value
			DoubleBarrierKnockOutCall call = new DoubleBarrierKnockOutCall(initialStockPrice, riskFreeRate, volatilityTerm,
					maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
					lowerBoundFactorA, lowerBoundExponentialDelta2);
		
			
		
		double[] iterationSteps = new double[numberOfIterationTimes+1];
		double[] importanceSamplingValues = new double[numberOfIterationTimes+1];
		double[] importanceSamplingSampleVariances = new double[numberOfIterationTimes+1];
		double[] importanceSamplingStandardErrors = new double[numberOfIterationTimes+1];
		double[] importanceSamplingErrors = new double[numberOfIterationTimes+1];
		double[] adaptiveImportanceSamplingValues = new double[numberOfIterationTimes+1];
		double[] adaptiveImportanceSamplingSampleVariances = new double[numberOfIterationTimes+1];
		double[] adaptiveImportanceSamplingStandardErrors = new double[numberOfIterationTimes+1];
		double[] adaptiveImportanceSamplingErrors = new double[numberOfIterationTimes+1];

		for(int i = 0; i < numberOfIterationTimes+1; i++) {
			iterationSteps[i] = i;
			
			
			//IS value
			AdaGradISCall parameterISCall = new AdaGradISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, i, epsilon);
			double[] eta = parameterISCall.getOptimalEta();
			DoubleBarrierKnockOutCallImportanceSampling callIS = new DoubleBarrierKnockOutCallImportanceSampling(numberOfSimulations, numberOfTimeSteps,
					randomNumberMatrix,
					initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
					maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
					lowerBoundExponentialDelta2, eta);
			double ISValue = callIS.getImportanceSamplingValue();
			double ISSampleVariance = callIS.getSampleVariance();
			double ISStandardError = callIS.getStandardError();
			
			importanceSamplingValues[i] = ISValue;	
			importanceSamplingSampleVariances[i] = ISSampleVariance;
			importanceSamplingStandardErrors[i] = ISStandardError;
			importanceSamplingErrors[i] = Math.abs(ISValue - call.getAnalyticValue())/call.getAnalyticValue();
		}
		for(int i = 0; i < numberOfIterationTimes+1; i++) {
			
			
			//IS value
			AdaGradISCall parameterISCall = new AdaGradISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon);
			double[] eta = parameterISCall.getOptimalEta();
			
			//AIS value
			AdaGradAISCall gradientDescentAISCall = new AdaGradAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
					numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, i, epsilon, eta);
			double[][][][] weightMatrixAIS = gradientDescentAISCall.getOptimalWeightMatrix();
			double[][][] weightMatrix1AIS = weightMatrixAIS[0];
			double[][][] weightMatrix2AIS = weightMatrixAIS[1];
			double[][][] weightMatrix3AIS = weightMatrixAIS[2];
			DoubleBarrierKnockOutCallAdaptiveImportanceSampling callAIS = new DoubleBarrierKnockOutCallAdaptiveImportanceSampling(numberOfSimulations, numberOfTimeSteps,
					numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
					initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
					maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
					lowerBoundExponentialDelta2, eta, weightMatrix1AIS, weightMatrix2AIS,
					weightMatrix3AIS);
			double AISValue = callAIS.getAdaptiveImportanceSamplingValue();
			double AISSampleVariance = callAIS.getSampleVariance();
			double AISStandardError = callAIS.getStandardError();
			adaptiveImportanceSamplingValues[i] = AISValue;	
			adaptiveImportanceSamplingSampleVariances[i] = AISSampleVariance;
			adaptiveImportanceSamplingStandardErrors[i] = AISStandardError;
			adaptiveImportanceSamplingErrors[i] = Math.abs(AISValue - call.getAnalyticValue())/call.getAnalyticValue();
		}
		
		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries series1 = new XYSeries("Estimation error under IS");
		for(int i = 0; i < numberOfIterationTimes+1; i++) {
			series1.add(i, importanceSamplingErrors[i]);
		}
		dataset.addSeries(series1);
		

		
		XYSeries series2 = new XYSeries("Estimation error under AIS");
		for(int i = 0; i < numberOfIterationTimes+1; i++) {
			series2.add(i, adaptiveImportanceSamplingErrors[i]);
		}
		dataset.addSeries(series2);
		

		

		
        return dataset;
    }

    private JFreeChart createChart(final XYDataset dataset) {

    	TimeDiscretizationWithEqualTimeStepSize time = new TimeDiscretizationWithEqualTimeStepSize(0, maturity, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		MersenneTwisterSequence randomnumber = new MersenneTwisterSequence(numberOfSimulations, timeSeries);
		double[][] randomNumberMatrix = randomnumber.getRandomNumberRealizations();
		
		//analytic value
			DoubleBarrierKnockOutCall call = new DoubleBarrierKnockOutCall(initialStockPrice, riskFreeRate, volatilityTerm,
					maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
					lowerBoundFactorA, lowerBoundExponentialDelta2);
		
			
		
		double[] iterationSteps = new double[numberOfIterationTimes];
		double[] importanceSamplingValues = new double[numberOfIterationTimes];
		double[] importanceSamplingSampleVariances = new double[numberOfIterationTimes];
		double[] importanceSamplingStandardErrors = new double[numberOfIterationTimes];
		double[] importanceSamplingErrors = new double[numberOfIterationTimes];
		double[] adaptiveImportanceSamplingValues = new double[numberOfIterationTimes];
		double[] adaptiveImportanceSamplingSampleVariances = new double[numberOfIterationTimes];
		double[] adaptiveImportanceSamplingStandardErrors = new double[numberOfIterationTimes];
		double[] adaptiveImportanceSamplingErrors = new double[numberOfIterationTimes];

		for(int i = 0; i < numberOfIterationTimes; i++) {
			iterationSteps[i] = i;
			
			
			//IS value
			AdaGradISCall parameterISCall = new AdaGradISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, i, epsilon);
			double[] eta = parameterISCall.getOptimalEta();
			DoubleBarrierKnockOutCallImportanceSampling callIS = new DoubleBarrierKnockOutCallImportanceSampling(numberOfSimulations, numberOfTimeSteps,
					randomNumberMatrix,
					initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
					maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
					lowerBoundExponentialDelta2, eta);
			double ISValue = callIS.getImportanceSamplingValue();
			double ISSampleVariance = callIS.getSampleVariance();
			double ISStandardError = callIS.getStandardError();
			
			importanceSamplingValues[i] = ISValue;	
			importanceSamplingSampleVariances[i] = ISSampleVariance;
			importanceSamplingStandardErrors[i] = ISStandardError;
			importanceSamplingErrors[i] = ISValue - call.getAnalyticValue();
		}
		for(int i = 0; i < numberOfIterationTimes; i++) {
			
			
			//IS value
			AdaGradISCall parameterISCall = new AdaGradISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon);
			double[] eta = parameterISCall.getOptimalEta();
			
			//AIS value
			AdaGradAISCall gradientDescentAISCall = new AdaGradAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
					numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, i, epsilon, eta);
			double[][][][] weightMatrixAIS = gradientDescentAISCall.getOptimalWeightMatrix();
			double[][][] weightMatrix1AIS = weightMatrixAIS[0];
			double[][][] weightMatrix2AIS = weightMatrixAIS[1];
			double[][][] weightMatrix3AIS = weightMatrixAIS[2];
			DoubleBarrierKnockOutCallAdaptiveImportanceSampling callAIS = new DoubleBarrierKnockOutCallAdaptiveImportanceSampling(numberOfSimulations, numberOfTimeSteps,
					numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
					initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
					maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
					lowerBoundExponentialDelta2, eta, weightMatrix1AIS, weightMatrix2AIS,
					weightMatrix3AIS);
			double AISValue = callAIS.getAdaptiveImportanceSamplingValue();
			double AISSampleVariance = callAIS.getSampleVariance();
			double AISStandardError = callAIS.getStandardError();
			adaptiveImportanceSamplingValues[i] = AISValue;	
			adaptiveImportanceSamplingSampleVariances[i] = AISSampleVariance;
			adaptiveImportanceSamplingStandardErrors[i] = AISStandardError;
			adaptiveImportanceSamplingErrors[i] = Math.abs(AISValue - call.getAnalyticValue())/call.getAnalyticValue();
		}
    	
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Double Knock-Out Option", 
                "Iteration step", 
                "Estimation error", 
                dataset, 
                PlotOrientation.VERTICAL,
                true, 
                true, 
                false
        );

        XYPlot plot = chart.getXYPlot();

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(true,true);

        
        
        renderer.setSeriesPaint(0, Color.BLACK);
       
        
    


        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.white);

        plot.setRangeGridlinesVisible(false);
        plot.setDomainGridlinesVisible(false);
        



        chart.setTitle(new TextTitle("Estimation Error of Double Knock-Out Option Valuation under IS and AIS method",
                        new Font("Serif", Font.BOLD, 24)
                )
        );
        

        
        return chart;
    }


}
