/**
 * 
 */
package master.thesis.experiments;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import master.thesis.adaptive.importance.sampling.DoubleBarrierKnockOutCallAdaptiveImportanceSampling;
import master.thesis.barrieroption.DoubleBarrierKnockOutCall;
import master.thesis.barrieroption.DoubleBarrierKnockOutPut;
import master.thesis.importance.sampling.DoubleBarrierKnockOutCallImportanceSampling;
import master.thesis.montecarlo.DoubleBarrierKnockOutCallMonteCarlo;
import master.thesis.montecarlo.DoubleBarrierKnockOutPutMonteCarlo;
import master.thesis.neural.network.GradientDescentAISCall;
import master.thesis.neural.network.GradientDescentISCall;
import master.thesis.randomnumber.MersenneTwisterSequence;
import master.thesis.timediscretization.TimeDiscretizationWithEqualTimeStepSize;
import net.finmath.plots.Plot;
import net.finmath.plots.Plots;

/**
 * @author QuanLiu
 *
 */
public class DoubleBarrierKnockOutOptionAnalyticValueExperiment {
	static double initialStockPrice = 100.0;
	static double riskFreeRate = 0.05;
	static double volatilityTerm = 0.1;
	
	static double maturity = 1.0;//set evaluation time to be 0
	static double strike = 100.0;
	
	static double upperBoundFactorB = 120.0;
	static double upperBoundExponentialDelta1 = 0.1;
	static double lowerBoundFactorA = 80.0;
	static double lowerBoundExponentialDelta2 = -0.1;
	
	static int numberOfSimulations = 2000;
	static int numberOfTimeSteps = 5;
	
	static int numberOfFirstHiddenLayerNeurons=10;
	static int numberOfSecondHiddenLayerNeurons=5;
	
	static double learningRate = 1.0;
	static int numberOfIterationTimes = 100;
	
	public static void main(String[] ards) throws Exception {
	    //valuationTest();
	    valuationPlot();
	}
	
	

	
	private static void valuationTest() throws Exception {
		TimeDiscretizationWithEqualTimeStepSize time = new TimeDiscretizationWithEqualTimeStepSize(0, maturity, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		MersenneTwisterSequence randomnumber = new MersenneTwisterSequence(numberOfSimulations, timeSeries);
		double[][] randomNumberMatrix = randomnumber.getRandomNumberRealizations();
		
		
		
		
		
		
		
		//analytic value
		DoubleBarrierKnockOutCall call = new DoubleBarrierKnockOutCall(initialStockPrice, riskFreeRate, volatilityTerm,
				maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
				lowerBoundFactorA, lowerBoundExponentialDelta2);
		System.out.println("Analytic value of double knock-out call is: " + call.getAnalyticValue());
		System.out.println();
		
		//monte carlo value
		DoubleBarrierKnockOutCallMonteCarlo callMC = new DoubleBarrierKnockOutCallMonteCarlo(initialStockPrice,riskFreeRate, volatilityTerm,
			    maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
				lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfSimulations, timeSeries,
				randomNumberMatrix);
		System.out.println("Monte Carlo value of double knock-out call is: " + callMC.getMonteCarloValue());
		System.out.println("Sample variance of double knock-out call under Monte Carlo is: " + callMC.getSampleVariance());
		System.out.println("Standard error of double knock-out call under Monte Carlo is: " + callMC.getStandardError());
		System.out.println();
		
		//IS value
		long timeStart = System.currentTimeMillis();
		GradientDescentISCall gradientDescentISCall = new GradientDescentISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
				numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes);
		double[][][] weightMatrixIS = gradientDescentISCall.getOptimalWeightMatrix();
		double[][] weightMatrix1IS = weightMatrixIS[0];
		double[][] weightMatrix2IS = weightMatrixIS[1];
		double[][] weightMatrix3IS = weightMatrixIS[2];
		long timeEnd = System.currentTimeMillis();
		double timeSec = (timeEnd-timeStart) / 1000.0;
		System.out.println("Time to find optimal natural parameters for call under IS...: " + timeSec + " sec.");
		DoubleBarrierKnockOutCallImportanceSampling callIS = new DoubleBarrierKnockOutCallImportanceSampling(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, weightMatrix1IS, weightMatrix2IS,
				weightMatrix3IS);
		System.out.println("Importance Sampling value of double knock-out call is: " + callIS.getImportanceSamplingValue());
		System.out.println("Sample variance of double knock-out call under Importance Sampling is: " + callIS.getSampleVariance());
		System.out.println("Standard error of double knock-out call under Importance Sampling is: " + callIS.getStandardError());
		System.out.println();
		

		//AIS value
		long timeStart1 = System.currentTimeMillis();
		GradientDescentAISCall gradientDescentAISCall = new GradientDescentAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
				numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes);
		double[][][][] weightMatrixAIS = gradientDescentAISCall.getOptimalWeightMatrix();
		double[][][] weightMatrix1AIS = weightMatrixAIS[0];
		double[][][] weightMatrix2AIS = weightMatrixAIS[1];
		double[][][] weightMatrix3AIS = weightMatrixAIS[2];
		long timeEnd1 = System.currentTimeMillis();
		double timeSec1 = (timeEnd1-timeStart1) / 1000.0;
		System.out.println("Time to find optimal natural parameters for call under forward AIS...: " + timeSec1 + " sec.");
		DoubleBarrierKnockOutCallAdaptiveImportanceSampling callAIS = new DoubleBarrierKnockOutCallAdaptiveImportanceSampling(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, weightMatrix1AIS, weightMatrix2AIS,
				weightMatrix3AIS);
		System.out.println("Forward Adaptive Importance Sampling value of double knock-out call is: " + callAIS.getAdaptiveImportanceSamplingValue());
		System.out.println("Sample variance of double knock-out call under Forward Adaptive Importance Sampling is: " + callAIS.getSampleVariance());
		System.out.println("Standard error of double knock-out call under Forward Adaptive Importance Sampling is: " + callAIS.getStandardError());

	}
		




	private static void valuationPlot() throws Exception {
		TimeDiscretizationWithEqualTimeStepSize time = new TimeDiscretizationWithEqualTimeStepSize(0, maturity, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		MersenneTwisterSequence randomnumber = new MersenneTwisterSequence(numberOfSimulations, timeSeries);
		double[][] randomNumberMatrix = randomnumber.getRandomNumberRealizations();
		
		final List<Double> iterationSteps = new ArrayList<Double>();
		final List<Double> analyticValues = new ArrayList<Double>();
		final List<Double> monteCarloValues = new ArrayList<Double>();
		final List<Double> monteCarloSampleVariances = new ArrayList<Double>();
		final List<Double> monteCarloStandardErrors = new ArrayList<Double>();
		final List<Double> importanceSamplingValues = new ArrayList<Double>();
		final List<Double> importanceSamplingSampleVariances = new ArrayList<Double>();
		final List<Double> importanceSamplingStandardErrors = new ArrayList<Double>();
		final List<Double> importanceSamplingErrors = new ArrayList<Double>();
		final List<Double> adaptiveImportanceSamplingValues = new ArrayList<Double>();	
		final List<Double> adaptiveImportanceSamplingSampleVariances = new ArrayList<Double>();
		final List<Double> adaptiveImportanceSamplingStandardErrors = new ArrayList<Double>();
		final List<Double> adaptiveImportanceSamplingErrors = new ArrayList<Double>();
		for(int i = 0; i < numberOfIterationTimes; i+=numberOfIterationTimes/10) {
			iterationSteps.add((double)i);
			//analytic value
			DoubleBarrierKnockOutCall call = new DoubleBarrierKnockOutCall(initialStockPrice, riskFreeRate, volatilityTerm,
					maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
					lowerBoundFactorA, lowerBoundExponentialDelta2);
			analyticValues.add(call.getAnalyticValue());	
			//monte carlo value
			DoubleBarrierKnockOutCallMonteCarlo callMC = new DoubleBarrierKnockOutCallMonteCarlo(initialStockPrice,riskFreeRate, volatilityTerm,
				    maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
					lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfSimulations, timeSeries,
					randomNumberMatrix);
			monteCarloValues.add(callMC.getMonteCarloValue());	
			monteCarloSampleVariances.add(callMC.getSampleVariance());
			monteCarloStandardErrors.add(callMC.getStandardError());
			//IS value
			GradientDescentISCall gradientDescentISCall = new GradientDescentISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
					numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, i);
			double[][][] weightMatrixIS = gradientDescentISCall.getOptimalWeightMatrix();
			double[][] weightMatrix1IS = weightMatrixIS[0];
			double[][] weightMatrix2IS = weightMatrixIS[1];
			double[][] weightMatrix3IS = weightMatrixIS[2];
			DoubleBarrierKnockOutCallImportanceSampling callIS = new DoubleBarrierKnockOutCallImportanceSampling(numberOfSimulations, numberOfTimeSteps,
					0, 0, randomNumberMatrix,
					initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
					maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
					lowerBoundExponentialDelta2, weightMatrix1IS, weightMatrix2IS,
					weightMatrix3IS);
			double ISValue = callIS.getImportanceSamplingValue();
			double ISSampleVariance = callIS.getSampleVariance();
			double ISStandardError = callIS.getStandardError();
			importanceSamplingValues.add(ISValue);	
			importanceSamplingSampleVariances.add(ISSampleVariance);
			importanceSamplingStandardErrors.add(ISStandardError);
			importanceSamplingErrors.add(ISValue - call.getAnalyticValue());
			
			
			
			//AIS value
			GradientDescentAISCall gradientDescentAISCall = new GradientDescentAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
					numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, i);
			double[][][][] weightMatrixAIS = gradientDescentAISCall.getOptimalWeightMatrix();
			double[][][] weightMatrix1AIS = weightMatrixAIS[0];
			double[][][] weightMatrix2AIS = weightMatrixAIS[1];
			double[][][] weightMatrix3AIS = weightMatrixAIS[2];
			DoubleBarrierKnockOutCallAdaptiveImportanceSampling callAIS = new DoubleBarrierKnockOutCallAdaptiveImportanceSampling(numberOfSimulations, numberOfTimeSteps,
					numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
					initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
					maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
					lowerBoundExponentialDelta2, weightMatrix1AIS, weightMatrix2AIS,
					weightMatrix3AIS);
			double AISValue = callAIS.getAdaptiveImportanceSamplingValue();
			double AISSampleVariance = callAIS.getSampleVariance();
			double AISStandardError = callAIS.getStandardError();
			adaptiveImportanceSamplingValues.add(AISValue);	
			adaptiveImportanceSamplingSampleVariances.add(AISSampleVariance);
			adaptiveImportanceSamplingStandardErrors.add(AISStandardError);
			adaptiveImportanceSamplingErrors.add(AISValue - call.getAnalyticValue());
		}
		
		final Plot plot = Plots.createScatter(iterationSteps, importanceSamplingErrors, 0.0, 0.1, 5)
				.setTitle("Estimation error under IS method for double knock-out call")
				.setXAxisLabel("iteration (training) times")
				.setYAxisLabel("estimation error")
				.setYRange(0.0, 0.5)
				.setYAxisNumberFormat(new DecimalFormat("0.0%"));

		plot.show();
		
		final Plot plot1 = Plots.createScatter(iterationSteps, adaptiveImportanceSamplingErrors, 0.0, 0.1, 5)
				.setTitle("Estimation error under AIS method for double knock-out call")
				.setXAxisLabel("iteration (training) times")
				.setYAxisLabel("estimation error")
				.setYRange(0.0, 0.1)
				.setYAxisNumberFormat(new DecimalFormat("0.0%"));

		plot1.show();
	}
}




	
	
	
		
		
		


	

