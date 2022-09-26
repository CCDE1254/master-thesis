/**
 * 
 */
package master.thesis.experiments;

import java.awt.BasicStroke;
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

import master.thesis.importance.sampling.DoubleBarrierKnockOutCallImportanceSampling;
import master.thesis.neural.network.OptimalParameterFinderISCall;
import master.thesis.randomnumber.MersenneTwisterSequence;
import master.thesis.timediscretization.TimeDiscretizationWithEqualTimeStepSize;
import master.thesis.underlying.UnderlyingPrice;
import master.thesis.underlying.UnderlyingPriceUnderISCall;

public class ImportanceSamplingPathPlot extends JFrame {

	static int numberOfSimulations = 50;
	static double stratTime = 0.0;
	static double endTime = 10.0;
	static int numberOfTimeSteps = 1000;
	static double initialStockPrice = 100;
	static double strike = 150;
	static double riskFreeRate = 0.02;
	static double volatilityTerm = 0.15;
	
	double upperBoundFactorB = 150;
	double upperBoundExponentialDelta1 = 0.06;
//	double upperBoundExponentialDelta1 = 0.06;
	double lowerBoundFactorA = 50;
	double lowerBoundExponentialDelta2 = 0.06;
//	double lowerBoundExponentialDelta2 = -0.06;
    int numberOfPossibleValues = 11;
    double minEta1 = -0.02;
	double maxEta1 = 0.02;
	double minEta2 = -0.55;
	double maxEta2 = -0.499;
	
	
    public static void main(String[] args) {

    	
    	
        SwingUtilities.invokeLater(() -> {
        	ImportanceSamplingPathPlot ex = new ImportanceSamplingPathPlot();
            ex.setVisible(true);
        });
    }
    
    public ImportanceSamplingPathPlot() {

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
    	TimeDiscretizationWithEqualTimeStepSize time = new TimeDiscretizationWithEqualTimeStepSize(stratTime, endTime, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		//generate random number sequence under given time series
		MersenneTwisterSequence randomnumber = new MersenneTwisterSequence(numberOfSimulations, timeSeries);
		double[][] randomNumberMatrix = randomnumber.getRandomNumberRealizations();
		

		OptimalParameterFinderISCall parameterISCall = new OptimalParameterFinderISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, endTime, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfPossibleValues , minEta1, maxEta1,
				minEta2, maxEta2);
		double[] eta = parameterISCall.getOptimalEta();
		UnderlyingPriceUnderISCall callIS = new UnderlyingPriceUnderISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, endTime, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, eta);
		
		double[][] underlyingPriceMatrix = callIS.getUnderlyingPriceMatrix();
		
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
    	TimeDiscretizationWithEqualTimeStepSize time = new TimeDiscretizationWithEqualTimeStepSize(stratTime, endTime, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		//generate random number sequence under given time series
		MersenneTwisterSequence randomnumber = new MersenneTwisterSequence(numberOfSimulations, timeSeries);
		double[][] randomNumberMatrix = randomnumber.getRandomNumberRealizations();
		

		OptimalParameterFinderISCall parameterISCall = new OptimalParameterFinderISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, endTime, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfPossibleValues , minEta1, maxEta1,
				minEta2, maxEta2);
		double[] eta = parameterISCall.getOptimalEta();
		UnderlyingPriceUnderISCall callIS = new UnderlyingPriceUnderISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, endTime, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, eta);
		
		double[][] underlyingPriceMatrix = callIS.getUnderlyingPriceMatrix();
    	
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Double Knock-Out Option", 
                "Time", 
                "Price", 
                dataset, 
                PlotOrientation.VERTICAL,
                true, 
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
        

        chart.getLegend().setFrame(BlockBorder.NONE);

        chart.setTitle(new TextTitle("Underlying Path Simulation Example of Double Knock-Out Option with Increasing Upper Bound and Decreasing Lower Bound Under Importance Sampling",
                        new Font("Serif", Font.BOLD, 24)
                )
        );
        
        System.out.println(eta[0]);
        System.out.println(eta[1]);
        System.out.println(callIS.getNormalParameters(eta)[0]);
        System.out.println(callIS.getNormalParameters(eta)[1]);
//        chart.setTitle(new TextTitle("Underlying Path Simulation Example of Double Knock-Out Option with Increasing Upper Bound and Decreasing Lower Bound Under Importance Sampling",
//                new Font("Serif", Font.BOLD, 24)
//        )
//        );


        
        return chart;
    }


}