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
import org.jfree.chart.block.BlockBorder;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import master.thesis.adaptive.importance.sampling.DoubleBarrierKnockOutCallAdaptiveImportanceSampling;
import master.thesis.gradient.descent.AdaGradAISCall;
import master.thesis.gradient.descent.AdaGradISCall;
import master.thesis.randomnumber.MersenneTwisterSequence;
import master.thesis.timediscretization.TimeDiscretizationWithEqualTimeStepSize;
import master.thesis.underlying.UnderlyingPriceUnderAISCall;
import master.thesis.underlying.UnderlyingPriceUnderISCall;

/**
 * @author QuanLiu
 *
 */
public class AdaptiveImportanceSamplingPathPlot extends JFrame {

	static double initialStockPrice = 100.0;
	static double riskFreeRate = 0.0;
	static double volatilityTerm = 0.15;
	
	static double maturity = 10.0;//set evaluation time to be 0
	static double strike = 80.0;
	
	static double upperBoundFactorB = 150.0;
	static double upperBoundExponentialDelta1 = 0.03;
	static double lowerBoundFactorA = 50.0;
	static double lowerBoundExponentialDelta2 = 0.03;
	
	static int numberOfSimulations = 10000;
	static int numberOfTimeSteps = 100;
	
	static int numberOfFirstHiddenLayerNeurons=10;
	static int numberOfSecondHiddenLayerNeurons=5;
	
	static double learningRate = 0.0000003;
	static int numberOfIterationTimes = 50;
	static double epsilon = 0.000000000000001;
	
    public static void main(String[] args) {

    	
    	
        SwingUtilities.invokeLater(() -> {
        	AdaptiveImportanceSamplingPathPlot ex = new AdaptiveImportanceSamplingPathPlot();
            ex.setVisible(true);
        });
    }
    
    public AdaptiveImportanceSamplingPathPlot() {

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

    	//generate array of time points
    	TimeDiscretizationWithEqualTimeStepSize time = new TimeDiscretizationWithEqualTimeStepSize(0.0, maturity, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		//generate random number sequence under given time series
		MersenneTwisterSequence randomnumber = new MersenneTwisterSequence(numberOfSimulations, timeSeries);
		double[][] randomNumberMatrix = randomnumber.getRandomNumberRealizations();
		

		AdaGradISCall parameterISCall = new AdaGradISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon);
		double[] eta = parameterISCall.getOptimalEta();
		
		
		//AIS value
		AdaGradAISCall gradientDescentAISCall = new AdaGradAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
				numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon, eta);
		double[][][][] weightMatrixAIS = gradientDescentAISCall.getOptimalWeightMatrix();
		double[][][] weightMatrix1AIS = weightMatrixAIS[0];
		double[][][] weightMatrix2AIS = weightMatrixAIS[1];
		double[][][] weightMatrix3AIS = weightMatrixAIS[2];
		UnderlyingPriceUnderAISCall callAIS = new UnderlyingPriceUnderAISCall(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, weightMatrix1AIS, weightMatrix2AIS,
				weightMatrix3AIS, eta);
		double[][] underlyingPriceMatrix = callAIS.getUnderlyingPriceMatrix();
		
		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries seriesU = new XYSeries("Upper Bound");
		for(int i = 0; i < timeSeries.length; i++) {
			seriesU.add(timeSeries[i], upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]));
		}
		dataset.addSeries(seriesU);
		
		XYSeries seriesL = new XYSeries("Lower Bound");
		for(int i = 0; i < timeSeries.length; i++) {
			seriesL.add(timeSeries[i], lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]));
		}
		dataset.addSeries(seriesL);
		
		
		for(int j = 0; j < numberOfSimulations; j++) {
			XYSeries series = new XYSeries("Path "+(j+1),true);
			for(int i = 0; i < timeSeries.length; i++) {
				series.add(timeSeries[i], underlyingPriceMatrix[i][j]);
			}
			dataset.addSeries(series);
		}
		

		
        return dataset;
    }

    private JFreeChart createChart(final XYDataset dataset) {

    	//generate array of time points
    	TimeDiscretizationWithEqualTimeStepSize time = new TimeDiscretizationWithEqualTimeStepSize(0.0, maturity, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		//generate random number sequence under given time series
		MersenneTwisterSequence randomnumber = new MersenneTwisterSequence(numberOfSimulations, timeSeries);
		double[][] randomNumberMatrix = randomnumber.getRandomNumberRealizations();
		

		AdaGradISCall parameterISCall = new AdaGradISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon);
		double[] eta = parameterISCall.getOptimalEta();
		
		
		//AIS value
		AdaGradAISCall gradientDescentAISCall = new AdaGradAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
				numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon, eta);
		double[][][][] weightMatrixAIS = gradientDescentAISCall.getOptimalWeightMatrix();
		double[][][] weightMatrix1AIS = weightMatrixAIS[0];
		double[][][] weightMatrix2AIS = weightMatrixAIS[1];
		double[][][] weightMatrix3AIS = weightMatrixAIS[2];
		UnderlyingPriceUnderAISCall callAIS = new UnderlyingPriceUnderAISCall(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, weightMatrix1AIS, weightMatrix2AIS,
				weightMatrix3AIS, eta);
		double[][] underlyingPriceMatrix = callAIS.getUnderlyingPriceMatrix();
    	
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Double Knock-Out Option", 
                "Time", 
                "Price", 
                dataset, 
                PlotOrientation.VERTICAL,
                false, 
                true, 
                false
        );

        XYPlot plot = chart.getXYPlot();

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(true,false);

        double[] upperBound = new double[timeSeries.length];
        for(int j = 0; j < timeSeries.length; j++) {
        	upperBound[j] = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[j]);
		}
        
        double[] lowerBound = new double[timeSeries.length];
        for(int j = 0; j < timeSeries.length; j++) {
        	lowerBound[j] = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[j]);
		}
        
        renderer.setSeriesPaint(0, Color.BLACK);
        renderer.setSeriesPaint(1, Color.BLACK);
        
        for(int j = 0; j < numberOfSimulations; j++) {
        	renderer.setSeriesPaint(j+2, Color.GREEN);
        	if(underlyingPriceMatrix[timeSeries.length-1][j]<strike) {
        		renderer.setSeriesPaint(j+2, Color.BLUE);
        	}
        	for(int i = 0; i < timeSeries.length; i++) {
    			if((underlyingPriceMatrix[i][j]>=upperBound[i]) || (underlyingPriceMatrix[i][j]<=lowerBound[i])) {
    				renderer.setSeriesPaint(j+2, Color.RED);
    			}
    		}
		}


        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.white);

        plot.setRangeGridlinesVisible(false);
        plot.setDomainGridlinesVisible(false);
        

//        chart.getLegend().setFrame(BlockBorder.NONE);

        chart.setTitle(new TextTitle("Adaptive Importance Sampling, N = " + numberOfTimeSteps + " , ?? = " + learningRate,
                        new Font("Serif", Font.BOLD, 40)
                )
        );
        

        
        return chart;
    }


}