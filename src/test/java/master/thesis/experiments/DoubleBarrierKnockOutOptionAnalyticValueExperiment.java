/**
 * 
 */
package master.thesis.experiments;

import master.thesis.adaptive.importance.sampling.DoubleBarrierKnockOutCallAdaptiveImportanceSampling;
import master.thesis.adaptive.importance.sampling.DoubleBarrierKnockOutPutAdaptiveImportanceSampling;
import master.thesis.barrieroption.DoubleBarrierKnockOutCall;
import master.thesis.barrieroption.DoubleBarrierKnockOutPut;
import master.thesis.importance.sampling.DoubleBarrierKnockOutCallImportanceSampling;
import master.thesis.importance.sampling.DoubleBarrierKnockOutPutImportanceSampling;
import master.thesis.montecarlo.DoubleBarrierKnockOutCallMonteCarlo;
import master.thesis.montecarlo.DoubleBarrierKnockOutPutMonteCarlo;
import master.thesis.neural.network.GradientDescentISPut;
import master.thesis.neural.network.GradientDescentBackwardAISCall;
import master.thesis.neural.network.GradientDescentBackwardAISPut;
import master.thesis.neural.network.GradientDescentForwardAISCall;
import master.thesis.neural.network.GradientDescentForwardAISPut;
import master.thesis.neural.network.GradientDescentISCall;
import master.thesis.randomnumber.MersenneTwisterSequence;
import master.thesis.timediscretization.TimeDiscretizationWithEqualTimeStepSize;

/**
 * @author QuanLiu
 *
 */
public class DoubleBarrierKnockOutOptionAnalyticValueExperiment {
	public static void main(String[] args) {
	double initialStockPrice = 100.0;
	double riskFreeRate = 0.05;
	double volatilityTerm = 0.1;
	
	double maturity = 1.0/2;//set evaluation time to be 0
	double strike = 100.0;
	
	double upperBoundFactorB = 115.0;
	double upperBoundExponentialDelta1 = 0.1;
	double lowerBoundFactorA = 85.0;
	double lowerBoundExponentialDelta2 = -0.1;
	
	int numberOfSimulations = 20;
	int numberOfTimeSteps = 20;
	
	int numberOfFirstHiddenLayerNeurons=10;
	int numberOfSecondHiddenLayerNeurons=5;
	
	double learningRate = 0.1;
	int numberOfIterationTimes = 200;
	

	
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
	double[] naturalParametersOfProposalDistributionCall = gradientDescentISCall.getOptimalNaturalParameterUnderIS();
	long timeEnd = System.currentTimeMillis();
	double timeSec = (timeEnd-timeStart) / 1000.0;
	System.out.println("Time to find optimal natural parameters for call under IS...: " + timeSec + " sec.");
	DoubleBarrierKnockOutCallImportanceSampling callIS = new DoubleBarrierKnockOutCallImportanceSampling(initialStockPrice,riskFreeRate, volatilityTerm,
		    maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
			lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfSimulations, timeSeries,
			randomNumberMatrix, naturalParametersOfProposalDistributionCall);
	System.out.println("Importance Sampling value of double knock-out call is: " + callIS.getImportanceSamplingValue());
	System.out.println("Sample variance of double knock-out call under Importance Sampling is: " + callIS.getSampleVariance());
	System.out.println("Standard error of double knock-out call under Importance Sampling is: " + callIS.getStandardError());
	System.out.println();
	
	//forward AIS value
	long timeStart1 = System.currentTimeMillis();
	GradientDescentForwardAISCall gradientDescentForwardAISCall = new GradientDescentForwardAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
			numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
			riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
			upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes);
	double[][] naturalParametersOfProposalDistributionsCallForward = gradientDescentForwardAISCall.getOptimalNaturalParameterUnderForwardAIS();
	long timeEnd1 = System.currentTimeMillis();
	double timeSec1 = (timeEnd1-timeStart1) / 1000.0;
	System.out.println("Time to find optimal natural parameters for call under forward AIS...: " + timeSec1 + " sec.");
	DoubleBarrierKnockOutCallAdaptiveImportanceSampling callForwardAIS = new DoubleBarrierKnockOutCallAdaptiveImportanceSampling(initialStockPrice,riskFreeRate, volatilityTerm,
		    maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
			lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfSimulations, timeSeries,
			randomNumberMatrix, naturalParametersOfProposalDistributionsCallForward);
	System.out.println("Forward Adaptive Importance Sampling value of double knock-out call is: " + callForwardAIS.getImportanceSamplingValue());
	System.out.println("Sample variance of double knock-out call under Forward Adaptive Importance Sampling is: " + callForwardAIS.getSampleVariance());
	System.out.println("Standard error of double knock-out call under Forward Adaptive Importance Sampling is: " + callForwardAIS.getStandardError());
	
	//backward AIS value
	long timeStart2 = System.currentTimeMillis();
	GradientDescentBackwardAISCall gradientDescentBackwardAISCall = new GradientDescentBackwardAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
			numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
			riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
			upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes);
	double[][] naturalParametersOfProposalDistributionsCallBackward = gradientDescentBackwardAISCall.getOptimalNaturalParameterUnderBackwardAIS();
	long timeEnd2 = System.currentTimeMillis();
	double timeSec2 = (timeEnd2-timeStart2) / 1000.0;
	System.out.println("Time to find optimal natural parameters for call under forward AIS...: " + timeSec2 + " sec.");
	DoubleBarrierKnockOutCallAdaptiveImportanceSampling callBackwardAIS = new DoubleBarrierKnockOutCallAdaptiveImportanceSampling(initialStockPrice,riskFreeRate, volatilityTerm,
		    maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
			lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfSimulations, timeSeries,
			randomNumberMatrix, naturalParametersOfProposalDistributionsCallBackward);
	System.out.println("Backward Adaptive Importance Sampling value of double knock-out call is: " + callBackwardAIS.getImportanceSamplingValue());
	System.out.println("Sample variance of double knock-out call under Backward Adaptive Importance Sampling is: " + callBackwardAIS.getSampleVariance());
	System.out.println("Standard error of double knock-out call under Backward Adaptive Importance Sampling is: " + callBackwardAIS.getStandardError());
	System.out.println();
	System.out.println();
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
	long timeStart3 = System.currentTimeMillis();
	GradientDescentISPut gradientDescentISPut = new GradientDescentISPut(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
			numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
			riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
			upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes);
	double[] naturalParametersOfProposalDistributionPut = gradientDescentISPut.getOptimalNaturalParameterUnderIS();
	long timeEnd3 = System.currentTimeMillis();
	double timeSec3 = (timeEnd3-timeStart3) / 1000.0;
	System.out.println("Time to find optimal natural parameters for put under IS...: " + timeSec3 + " sec.");
	DoubleBarrierKnockOutPutImportanceSampling putIS = new DoubleBarrierKnockOutPutImportanceSampling(initialStockPrice,riskFreeRate, volatilityTerm,
		    maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
			lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfSimulations, timeSeries,
			randomNumberMatrix, naturalParametersOfProposalDistributionPut);
	System.out.println("Importance Sampling value of double knock-out put is: " + putIS.getImportanceSamplingValue());
	System.out.println("Sample variance of double knock-out put under Importance Sampling is: " + putIS.getSampleVariance());
	System.out.println("Standard error of double knock-out put under Importance Sampling is: " + putIS.getStandardError());
	System.out.println();
	
	//forward AIS value
		long timeStart4 = System.currentTimeMillis();
		GradientDescentForwardAISPut gradientDescentForwardAISPut = new GradientDescentForwardAISPut(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
				numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes);
		double[][] naturalParametersOfProposalDistributionsPutForward = gradientDescentForwardAISPut.getOptimalNaturalParameterUnderForwardAIS();
		long timeEnd4 = System.currentTimeMillis();
		double timeSec4 = (timeEnd4-timeStart4) / 1000.0;
		System.out.println("Time to find optimal natural parameters for put under forward AIS...: " + timeSec4 + " sec.");
		DoubleBarrierKnockOutPutAdaptiveImportanceSampling putForwardAIS = new DoubleBarrierKnockOutPutAdaptiveImportanceSampling(initialStockPrice,riskFreeRate, volatilityTerm,
			    maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
				lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfSimulations, timeSeries,
				randomNumberMatrix, naturalParametersOfProposalDistributionsPutForward);
		System.out.println("Forward Adaptive Importance Sampling value of double knock-out put is: " + putForwardAIS.getImportanceSamplingValue());
		System.out.println("Sample variance of double knock-out put under Forward Adaptive Importance Sampling is: " + putForwardAIS.getSampleVariance());
		System.out.println("Standard error of double knock-out put under Forward Adaptive Importance Sampling is: " + putForwardAIS.getStandardError());
		
		//backward AIS value
		long timeStart5 = System.currentTimeMillis();
		GradientDescentBackwardAISPut gradientDescentBackwardAISPut = new GradientDescentBackwardAISPut(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
				numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2, learningRate, numberOfIterationTimes);
		double[][] naturalParametersOfProposalDistributionsPutBackward = gradientDescentBackwardAISPut.getOptimalNaturalParameterUnderBackwardAIS();
		long timeEnd5 = System.currentTimeMillis();
		double timeSec5 = (timeEnd5-timeStart5) / 1000.0;
		System.out.println("Time to find optimal natural parameters for put under forward AIS...: " + timeSec5 + " sec.");
		DoubleBarrierKnockOutPutAdaptiveImportanceSampling putBackwardAIS = new DoubleBarrierKnockOutPutAdaptiveImportanceSampling(initialStockPrice,riskFreeRate, volatilityTerm,
			    maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
				lowerBoundFactorA, lowerBoundExponentialDelta2, numberOfSimulations, timeSeries,
				randomNumberMatrix, naturalParametersOfProposalDistributionsPutBackward);
		System.out.println("Backward Adaptive Importance Sampling value of double knock-out put is: " + putBackwardAIS.getImportanceSamplingValue());
		System.out.println("Sample variance of double knock-out put under Backward Adaptive Importance Sampling is: " + putBackwardAIS.getSampleVariance());
		System.out.println("Standard error of double knock-out put under Backward Adaptive Importance Sampling is: " + putBackwardAIS.getStandardError());
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		
		for(int i = 0; i < naturalParametersOfProposalDistributionsCallForward.length; i++) {
			for(int j = 0; j < naturalParametersOfProposalDistributionsCallForward[0].length; j++) {
				System.out.println(naturalParametersOfProposalDistributionsCallForward[i][j]);
			}
			System.out.println();
		}
	}
	
}
