/**
 * 
 */
package master.thesis.importance.sampling;

import master.thesis.underlying.UnderlyingPriceUnderProposalDistribution;

/**
 * @author QuanLiu
 *
 */
public class DoubleBarrierKnockOutPutImportanceSampling {
	double initialStockPrice;
	double riskFreeRate;
	double volatilityTerm;
	
	double maturity;//set evaluation time to be 0
	double strike;
	
	double upperBoundFactorB;
	double upperBoundExponentialDelta1;
	double lowerBoundFactorA;
	double lowerBoundExponentialDelta2;
	
	int numberOfSimulations;
	double[] timeSeries;
	double[][] randomNumberMatrix;
	double[] naturalParametersOfProposalDistribution;
	
		
	
	
	public DoubleBarrierKnockOutPutImportanceSampling(double initialStockPrice, double riskFreeRate,
			double volatilityTerm, double maturity, double strike, double upperBoundFactorB,
			double upperBoundExponentialDelta1, double lowerBoundFactorA, double lowerBoundExponentialDelta2,
			int numberOfSimulations, double[] timeSeries, double[][] randomNumberMatrix,
			double[] naturalParametersOfProposalDistribution) {
		super();
		this.initialStockPrice = initialStockPrice;
		this.riskFreeRate = riskFreeRate;
		this.volatilityTerm = volatilityTerm;
		this.maturity = maturity;
		this.strike = strike;
		this.upperBoundFactorB = upperBoundFactorB;
		this.upperBoundExponentialDelta1 = upperBoundExponentialDelta1;
		this.lowerBoundFactorA = lowerBoundFactorA;
		this.lowerBoundExponentialDelta2 = lowerBoundExponentialDelta2;
		this.numberOfSimulations = numberOfSimulations;
		this.timeSeries = timeSeries;
		this.randomNumberMatrix = randomNumberMatrix;
		this.naturalParametersOfProposalDistribution = naturalParametersOfProposalDistribution;
	}
	
	
	public double[][] getIncrementsMatrixUnderProposalDistribution() {
		UnderlyingPriceUnderProposalDistribution underlyingPrice = new UnderlyingPriceUnderProposalDistribution(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix,naturalParametersOfProposalDistribution);
		double[][] incrementsMatrixUnderProposalDistribution = underlyingPrice.getIncrementsMatrixUnderProposalDistribution();
		return incrementsMatrixUnderProposalDistribution;
	}
	
	public double[][] getUnderlyingPriceMatrixUnderProposalDistribution() {
		UnderlyingPriceUnderProposalDistribution underlyingPrice = new UnderlyingPriceUnderProposalDistribution(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix,naturalParametersOfProposalDistribution);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
		return underlyingPriceMatrix;
	}
	
	public double[] getDiscountedPayoffsUnderProposalDistribution() {
		double[][] underlyingPriceMatrix = getUnderlyingPriceMatrixUnderProposalDistribution();
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
			discountedPayoffs[j] = Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*Math.exp(-riskFreeRate*maturity)*payoffValid;
			
		}
		return discountedPayoffs;
	}
	

	public double[] getAdjustedDiscountedPayoffs() {
		double[][] incrementsMatrixUnderProposalDistribution = getIncrementsMatrixUnderProposalDistribution();
		double[] discountedPayoffs = getDiscountedPayoffsUnderProposalDistribution();
		double[] adjustedDiscountedPayoffs = new double[numberOfSimulations];
		for(int j = 0; j < numberOfSimulations; j++) {
			double adjustFactor = 1.0;
			for(int i = 0; i < timeSeries.length-1; i++) {
				double aeta = (1.0/Math.sqrt(-2.0*naturalParametersOfProposalDistribution[1]))*Math.exp(-naturalParametersOfProposalDistribution[0]*naturalParametersOfProposalDistribution[0]/4.0/naturalParametersOfProposalDistribution[1]);
				double likelihoodRatio = (1.0/aeta) * Math.exp(-naturalParametersOfProposalDistribution[0]*incrementsMatrixUnderProposalDistribution[i][j] - (1.0/2 + naturalParametersOfProposalDistribution[1])*Math.pow(incrementsMatrixUnderProposalDistribution[i][j], 2) );
			    adjustFactor *= likelihoodRatio;
			}
			adjustedDiscountedPayoffs[j] = discountedPayoffs[j] * adjustFactor;
		}
		return adjustedDiscountedPayoffs;
	}
	
	public double getImportanceSamplingValue() {
		double[] adjustedDiscountedPayoffs = getAdjustedDiscountedPayoffs();
		double sumOfPayoffs = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfPayoffs += adjustedDiscountedPayoffs[j];
		}
		return sumOfPayoffs/numberOfSimulations;
	}
	

	public double getStandardError() {
		double[] adjustedDiscountedPayoffs = getAdjustedDiscountedPayoffs();
		double meanValue = getImportanceSamplingValue();
		double sumOfVariance = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfVariance += Math.pow(adjustedDiscountedPayoffs[j] - meanValue, 2.0);
		}
		return Math.sqrt(sumOfVariance/numberOfSimulations)/Math.sqrt(numberOfSimulations);
	}
	
	public double getSampleVariance() {
		double[] adjustedDiscountedPayoffs = getAdjustedDiscountedPayoffs();
		double meanValue = getImportanceSamplingValue();
		double sumOfVariance = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfVariance += Math.pow(adjustedDiscountedPayoffs[j] - meanValue, 2.0);
		}
		return sumOfVariance/numberOfSimulations;
	}
}
