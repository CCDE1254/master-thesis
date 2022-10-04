/**
 * 
 */
package master.thesis.experiments;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import master.thesis.adaptive.importance.sampling.DoubleBarrierKnockOutCallAdaptiveImportanceSampling;
import master.thesis.adaptive.importance.sampling.DoubleBarrierKnockOutPutAdaptiveImportanceSampling;
import master.thesis.barrieroption.DoubleBarrierKnockOutCall;
import master.thesis.barrieroption.DoubleBarrierKnockOutPut;
import master.thesis.gradient.descent.AdaGradAISCall;
import master.thesis.gradient.descent.AdaGradAISPut;
import master.thesis.gradient.descent.AdaGradISCall;
import master.thesis.gradient.descent.AdaGradISPut;
import master.thesis.importance.sampling.DoubleBarrierKnockOutCallImportanceSampling;
import master.thesis.importance.sampling.DoubleBarrierKnockOutPutImportanceSampling;
import master.thesis.montecarlo.DoubleBarrierKnockOutCallMonteCarlo;
import master.thesis.montecarlo.DoubleBarrierKnockOutPutMonteCarlo;
import master.thesis.randomnumber.MersenneTwisterSequence;
import master.thesis.timediscretization.TimeDiscretizationWithEqualTimeStepSize;
import net.finmath.plots.Plot;
import net.finmath.plots.Plots;

/**
 * @author QuanLiu
 *
 */
public class DoubleBarrierKnockOutOptionValuationExperiment {
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
	
	public static void main(String[] ards) throws Exception {
	    valuationTest();
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
		AdaGradISCall parameterISCall = new AdaGradISCall(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon);
		double[] eta = parameterISCall.getOptimalEta();
		long timeEnd = System.currentTimeMillis();
		double timeSec = (timeEnd-timeStart) / 1000.0;
		System.out.println("Time to find optimal natural parameters for call under IS...: " + timeSec + " sec.");
		DoubleBarrierKnockOutCallImportanceSampling callIS = new DoubleBarrierKnockOutCallImportanceSampling(numberOfSimulations, numberOfTimeSteps,
				randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, eta);
		System.out.println("Importance Sampling value of double knock-out call is: " + callIS.getImportanceSamplingValue());
		System.out.println("Sample variance of double knock-out call under Importance Sampling is: " + callIS.getSampleVariance());
		System.out.println("Standard error of double knock-out call under Importance Sampling is: " + callIS.getStandardError());
		System.out.println();

		//AIS value
		long timeStart1 = System.currentTimeMillis();
		AdaGradAISCall gradientDescentAISCall = new AdaGradAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
				numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon, eta);
		double[][][][] weightMatrixAIS = gradientDescentAISCall.getOptimalWeightMatrix();
		double[][][] weightMatrix1AIS = weightMatrixAIS[0];
		double[][][] weightMatrix2AIS = weightMatrixAIS[1];
		double[][][] weightMatrix3AIS = weightMatrixAIS[2];
		long timeEnd1 = System.currentTimeMillis();
		double timeSec1 = (timeEnd1-timeStart1) / 1000.0;
		System.out.println("Time to find optimal natural parameters for call under AIS...: " + timeSec1 + " sec.");
		DoubleBarrierKnockOutCallAdaptiveImportanceSampling callAIS = new DoubleBarrierKnockOutCallAdaptiveImportanceSampling(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, eta,weightMatrix1AIS, weightMatrix2AIS,
				weightMatrix3AIS);
		System.out.println("Adaptive Importance Sampling value of double knock-out call is: " + callAIS.getAdaptiveImportanceSamplingValue());
		System.out.println("Sample variance of double knock-out call under Adaptive Importance Sampling is: " + callAIS.getSampleVariance());
		System.out.println("Standard error of double knock-out call under Adaptive Importance Sampling is: " + callAIS.getStandardError());
		System.out.println();
		System.out.println();
		
		
		//analytic value
		DoubleBarrierKnockOutPut put = new DoubleBarrierKnockOutPut(initialStockPrice, riskFreeRate, volatilityTerm,
				maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
				lowerBoundFactorA, lowerBoundExponentialDelta2);
		System.out.println("Analytic value of double knock-out put is: " + put.getAnalyticValue());
		System.out.println();
		
		//monte carlo value
		DoubleBarrierKnockOutPutMonteCarlo putMC = new DoubleBarrierKnockOutPutMonteCarlo(initialStockPrice,riskFreeRate, volatilityTerm,
			    maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
				lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfSimulations, timeSeries,
				randomNumberMatrix);
		System.out.println("Monte Carlo value of double knock-out put is: " + putMC.getMonteCarloValue());
		System.out.println("Sample variance of double knock-out put under Monte Carlo is: " + putMC.getSampleVariance());
		System.out.println("Standard error of double knock-out put under Monte Carlo is: " + putMC.getStandardError());
		System.out.println();
		
		//IS value
		long timeStart2 = System.currentTimeMillis();
		AdaGradISPut parameterISPut = new AdaGradISPut(numberOfSimulations, numberOfTimeSteps,  randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon);
		double[] eta1 = parameterISCall.getOptimalEta();
		long timeEnd2 = System.currentTimeMillis();
		double timeSec2 = (timeEnd2-timeStart2) / 1000.0;
		System.out.println("Time to find optimal natural parameters for put under IS...: " + timeSec2 + " sec.");
		DoubleBarrierKnockOutPutImportanceSampling putIS = new DoubleBarrierKnockOutPutImportanceSampling(numberOfSimulations, numberOfTimeSteps,
				randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, eta1);
		System.out.println("Importance Sampling value of double knock-out put is: " + putIS.getImportanceSamplingValue());
		System.out.println("Sample variance of double knock-out put under Importance Sampling is: " + putIS.getSampleVariance());
		System.out.println("Standard error of double knock-out put under Importance Sampling is: " + putIS.getStandardError());
		System.out.println();

		//AIS value
		long timeStart3 = System.currentTimeMillis();
		AdaGradAISPut gradientDescentAISPut = new AdaGradAISPut(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
				numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes, epsilon, eta);
		double[][][][] weightMatrixAIS1 = gradientDescentAISPut.getOptimalWeightMatrix();
		double[][][] weightMatrix1AIS1 = weightMatrixAIS1[0];
		double[][][] weightMatrix2AIS1 = weightMatrixAIS1[1];
		double[][][] weightMatrix3AIS1 = weightMatrixAIS1[2];
		long timeEnd3 = System.currentTimeMillis();
		double timeSec3 = (timeEnd3-timeStart3) / 1000.0;
		System.out.println("Time to find optimal natural parameters for put under AIS...: " + timeSec3 + " sec.");
		DoubleBarrierKnockOutPutAdaptiveImportanceSampling putAIS = new DoubleBarrierKnockOutPutAdaptiveImportanceSampling(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, eta,weightMatrix1AIS1, weightMatrix2AIS1,
				weightMatrix3AIS1);
		System.out.println("Adaptive Importance Sampling value of double knock-out put is: " + putAIS.getAdaptiveImportanceSamplingValue());
		System.out.println("Sample variance of double knock-out put under Adaptive Importance Sampling is: " + putAIS.getSampleVariance());
		System.out.println("Standard error of double knock-out put under Adaptive Importance Sampling is: " + putAIS.getStandardError());
	}
		

}




	
	
	
		
		
		


	

