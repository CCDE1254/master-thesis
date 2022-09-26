/**
 * 
 */
package master.thesis.adaptive.importance.sampling;

import master.thesis.neural.network.NeuralNetworkAISCall;
import master.thesis.underlying.UnderlyingPriceUnderAISCall;
import net.finmath.functions.NormalDistribution;

/**
 * @author QuanLiu
 *
 */
public class DoubleBarrierKnockOutCallAdaptiveImportanceSampling {
	int numberOfSimulations;
	int numberOfTimeSteps;
	int numberOfFirstHiddenLayerNeurons;
	int numberOfSecondHiddenLayerNeurons;
	double[][] randomNumberMatrix;
	double initialStockPrice;
	double riskFreeRate;
	double volatilityTerm;
	
	double[] timeSeries;
	double strike;
	double maturity;
	
	double upperBoundFactorB;
	double upperBoundExponentialDelta1;
	double lowerBoundFactorA;
	double lowerBoundExponentialDelta2;
	double[] etaUnderIS;
	double[][][] weightMatrix1;
	double[][][] weightMatrix2;
	double[][][] weightMatrix3;
	

	public DoubleBarrierKnockOutCallAdaptiveImportanceSampling(int numberOfSimulations, int numberOfTimeSteps,
			int numberOfFirstHiddenLayerNeurons, int numberOfSecondHiddenLayerNeurons, double[][] randomNumberMatrix,
			double initialStockPrice, double riskFreeRate, double volatilityTerm, double[] timeSeries, double strike,
			double maturity, double upperBoundFactorB, double upperBoundExponentialDelta1, double lowerBoundFactorA,
			double lowerBoundExponentialDelta2, double[] etaUnderIS, double[][][] weightMatrix1,
			double[][][] weightMatrix2, double[][][] weightMatrix3) {
		super();
		this.numberOfSimulations = numberOfSimulations;
		this.numberOfTimeSteps = numberOfTimeSteps;
		this.numberOfFirstHiddenLayerNeurons = numberOfFirstHiddenLayerNeurons;
		this.numberOfSecondHiddenLayerNeurons = numberOfSecondHiddenLayerNeurons;
		this.randomNumberMatrix = randomNumberMatrix;
		this.initialStockPrice = initialStockPrice;
		this.riskFreeRate = riskFreeRate;
		this.volatilityTerm = volatilityTerm;
		this.timeSeries = timeSeries;
		this.strike = strike;
		this.maturity = maturity;
		this.upperBoundFactorB = upperBoundFactorB;
		this.upperBoundExponentialDelta1 = upperBoundExponentialDelta1;
		this.lowerBoundFactorA = lowerBoundFactorA;
		this.lowerBoundExponentialDelta2 = lowerBoundExponentialDelta2;
		this.etaUnderIS = etaUnderIS;
		this.weightMatrix1 = weightMatrix1;
		this.weightMatrix2 = weightMatrix2;
		this.weightMatrix3 = weightMatrix3;
	}

	public double[][] getIncrementsMatrixUnderProposalDistributions() {
		UnderlyingPriceUnderAISCall underlyingPrice = new UnderlyingPriceUnderAISCall(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, weightMatrix1, weightMatrix2,
				weightMatrix3);
		double[][] incrementsMatrixUnderProposalDistribution = underlyingPrice.getIncrementsMatrixUnderAIS();
		return incrementsMatrixUnderProposalDistribution;
	}
	
	public double[][] getUnderlyingPriceMatrixUnderProposalDistributions() {
		UnderlyingPriceUnderAISCall underlyingPrice = new UnderlyingPriceUnderAISCall(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, weightMatrix1, weightMatrix2,
				weightMatrix3);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
		return underlyingPriceMatrix;
	}
	
	public double[] getDiscountedPayoffsUnderProposalDistributions() {
		double[][] incrementsMatrixUnderProposalDistribution = getIncrementsMatrixUnderProposalDistributions();
		double[][] underlyingPriceMatrix = getUnderlyingPriceMatrixUnderProposalDistributions();
		double[] discountedPayoffs = new double[numberOfSimulations];
		for(int j = 0; j < numberOfSimulations; j++) {
			double payoffValid = 1.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][j]>=upperBound) || (underlyingPriceMatrix[i][j]<=lowerBound)) {
					payoffValid = 0.0;
				}
			}
			NeuralNetworkAISCall neural = new NeuralNetworkAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
					numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2);
			double[][] inputVector = new double[numberOfTimeSteps - 1][timeSeries.length-1];
			for(int k = 1; k < numberOfTimeSteps - 1; k++) {
				inputVector[k][0] = 1.0;
				for(int i = 1; i < timeSeries.length-1; i++) {
					inputVector[k][i] = NormalDistribution.inverseCumulativeDistribution(randomNumberMatrix[i-1][j]);
				}
			}
			double[][] eta = neural.getOutput(inputVector, weightMatrix1, weightMatrix2, weightMatrix3);
			double productOfDensity = 1.0;
			productOfDensity *= Math.pow(-2.0*etaUnderIS[1], 1.0/2.0)*Math.exp(etaUnderIS[0]*etaUnderIS[1]/4.0/etaUnderIS[1])*Math.exp(-etaUnderIS[0]*incrementsMatrixUnderProposalDistribution[0][j])
					*Math.exp(-(1.0/2.0+etaUnderIS[1])*incrementsMatrixUnderProposalDistribution[0][j]*incrementsMatrixUnderProposalDistribution[0][j]);
			for(int i = 0; i < numberOfTimeSteps-1; i++) {
				productOfDensity *= Math.pow(-2.0*eta[i][1], 1.0/2.0)*Math.exp(eta[i][0]*eta[i][1]/4.0/eta[i][1])*Math.exp(-eta[i][0]*incrementsMatrixUnderProposalDistribution[i+1][j])
						*Math.exp(-(1.0/2.0+eta[i][1])*incrementsMatrixUnderProposalDistribution[i+1][j]*incrementsMatrixUnderProposalDistribution[i+1][j]);
			}
			discountedPayoffs[j] = Math.max((underlyingPriceMatrix[timeSeries.length-1][j] - strike),0.0)*Math.exp(-riskFreeRate*maturity)*payoffValid
					*productOfDensity;
			
		}
		return discountedPayoffs;
	}
	

	
	public double getAdaptiveImportanceSamplingValue() {
		double[] adjustedDiscountedPayoffs = getDiscountedPayoffsUnderProposalDistributions();
		double sumOfPayoffs = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfPayoffs += adjustedDiscountedPayoffs[j];
		}
		return sumOfPayoffs/numberOfSimulations;
	}
	

	public double getStandardError() {
		double[] adjustedDiscountedPayoffs = getDiscountedPayoffsUnderProposalDistributions();
		double meanValue = getAdaptiveImportanceSamplingValue();
		double sumOfVariance = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfVariance += Math.pow(adjustedDiscountedPayoffs[j] - meanValue, 2.0);
		}
		return Math.sqrt(sumOfVariance/numberOfSimulations)/Math.sqrt(numberOfSimulations);
	}
	
	public double getSampleVariance() {
		double[] adjustedDiscountedPayoffs = getDiscountedPayoffsUnderProposalDistributions();
		double meanValue = getAdaptiveImportanceSamplingValue();
		double sumOfVariance = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfVariance += Math.pow(adjustedDiscountedPayoffs[j] - meanValue, 2.0);
		}
		return sumOfVariance/numberOfSimulations;
	}
}
