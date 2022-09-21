/**
 * 
 */
package master.thesis.importance.sampling;

import master.thesis.neural.network.NeuralNetworkISCall;
import master.thesis.underlying.UnderlyingPriceUnderISCall;
import net.finmath.functions.NormalDistribution;

/**
 * @author QuanLiu
 *
 */
public class DoubleBarrierKnockOutCallImportanceSampling {
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
	
	double[][] weightMatrix1;
	double[][] weightMatrix2;
	double[][] weightMatrix3;
	
	public DoubleBarrierKnockOutCallImportanceSampling(int numberOfSimulations, int numberOfTimeSteps,
			int numberOfFirstHiddenLayerNeurons, int numberOfSecondHiddenLayerNeurons, double[][] randomNumberMatrix,
			double initialStockPrice, double riskFreeRate, double volatilityTerm, double[] timeSeries, double strike,
			double maturity, double upperBoundFactorB, double upperBoundExponentialDelta1, double lowerBoundFactorA,
			double lowerBoundExponentialDelta2, double[][] weightMatrix1, double[][] weightMatrix2,
			double[][] weightMatrix3) {
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
		this.weightMatrix1 = weightMatrix1;
		this.weightMatrix2 = weightMatrix2;
		this.weightMatrix3 = weightMatrix3;
	}

	public double[][] getIncrementsMatrixUnderProposalDistribution() {
		UnderlyingPriceUnderISCall underlyingPrice = new UnderlyingPriceUnderISCall(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, weightMatrix1, weightMatrix2,
				weightMatrix3);;
		double[][] incrementsMatrixUnderProposalDistribution = underlyingPrice.getIncrementsMatrixUnderIS();
		return incrementsMatrixUnderProposalDistribution;
	}
	
	public double[][] getUnderlyingPriceMatrixUnderProposalDistribution() {
		UnderlyingPriceUnderISCall underlyingPrice = new UnderlyingPriceUnderISCall(numberOfSimulations, numberOfTimeSteps,
				numberOfFirstHiddenLayerNeurons, numberOfSecondHiddenLayerNeurons, randomNumberMatrix,
				initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
				maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
				lowerBoundExponentialDelta2, weightMatrix1, weightMatrix2,
				weightMatrix3);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
		return underlyingPriceMatrix;
	}
	
	public double[] getDiscountedPayoffsUnderProposalDistribution() {
		double[][] incrementsMatrixUnderProposalDistribution = getIncrementsMatrixUnderProposalDistribution();
		double[][] underlyingPriceMatrix = getUnderlyingPriceMatrixUnderProposalDistribution();
		double[] discountedPayoffs = new double[numberOfSimulations];
		for(int j = 0; j < numberOfSimulations; j++) {
			double payoffValid = 1.0;
			double sumOfRealizations = 0.0;
			double sumOfSquaredRealizations = 0.0;
			for(int i = 0; i < timeSeries.length-1; i++) {
				sumOfRealizations += incrementsMatrixUnderProposalDistribution[i][j];
				sumOfSquaredRealizations += incrementsMatrixUnderProposalDistribution[i][j]*incrementsMatrixUnderProposalDistribution[i][j];
			}
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][j]>=upperBound) || (underlyingPriceMatrix[i][j]<=lowerBound)) {
					payoffValid = 0.0;
				}
			}
			NeuralNetworkISCall neural = new NeuralNetworkISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
					numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2);
			double[] inputVector = new double[timeSeries.length];
			inputVector[0] = 1.0;
			for(int i = 1; i < timeSeries.length; i++) {
				inputVector[i] = NormalDistribution.inverseCumulativeDistribution(randomNumberMatrix[i-1][j]);
			}
			double eta1 = neural.getOutput(inputVector, weightMatrix1, weightMatrix2, weightMatrix3)[0];
			double eta2 = neural.getOutput(inputVector, weightMatrix1, weightMatrix2, weightMatrix3)[1];
			discountedPayoffs[j] = Math.max((underlyingPriceMatrix[timeSeries.length-1][j] - strike),0.0)*Math.exp(-riskFreeRate*maturity)*payoffValid
					*Math.pow(-2.0*eta2, numberOfTimeSteps/2.0)*Math.exp(numberOfTimeSteps*eta1*eta1/4.0/eta2)*Math.exp(-eta1*sumOfRealizations)*Math.exp(-(1.0/2.0+eta2)*sumOfSquaredRealizations);
			
		}
		return discountedPayoffs;
	}
	

	
	public double getImportanceSamplingValue() {
		double[] adjustedDiscountedPayoffs = getDiscountedPayoffsUnderProposalDistribution();
		double sumOfPayoffs = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfPayoffs += adjustedDiscountedPayoffs[j];
		}
		return sumOfPayoffs/numberOfSimulations;
	}
	

	public double getStandardError() {
		double[] adjustedDiscountedPayoffs = getDiscountedPayoffsUnderProposalDistribution();
		double meanValue = getImportanceSamplingValue();
		double sumOfVariance = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfVariance += Math.pow(adjustedDiscountedPayoffs[j] - meanValue, 2.0);
		}
		return Math.sqrt(sumOfVariance/numberOfSimulations)/Math.sqrt(numberOfSimulations);
	}
	
	public double getSampleVariance() {
		double[] adjustedDiscountedPayoffs = getDiscountedPayoffsUnderProposalDistribution();
		double meanValue = getImportanceSamplingValue();
		double sumOfVariance = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfVariance += Math.pow(adjustedDiscountedPayoffs[j] - meanValue, 2.0);
		}
		return sumOfVariance/numberOfSimulations;
	}

}
