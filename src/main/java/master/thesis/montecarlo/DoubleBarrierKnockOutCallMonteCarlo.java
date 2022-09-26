/**
 * 
 */
package master.thesis.montecarlo;

import master.thesis.underlying.UnderlyingPrice;

/**
 * @author QuanLiu
 *
 */
public class DoubleBarrierKnockOutCallMonteCarlo {

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
	
	
	
	public DoubleBarrierKnockOutCallMonteCarlo(double initialStockPrice, double riskFreeRate, double volatilityTerm,
			double maturity, double strike, double upperBoundFactorB, double upperBoundExponentialDelta1,
			double lowerBoundFactorA, double lowerBoundExponentialDelta2, int numberOfSimulations, double[] timeSeries,
			double[][] randomNumberMatrix) {
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
	}
	
	
	public double[] getDiscountedPayoffs() {
		UnderlyingPrice underlyingPrice = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
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
			discountedPayoffs[j] = Math.max((underlyingPriceMatrix[timeSeries.length-1][j] - strike),0.0)*Math.exp(-riskFreeRate*maturity)*payoffValid;
		}
		return discountedPayoffs;
	}
	
	public double getMonteCarloValue() {
		double[] discountedPayoffs = getDiscountedPayoffs();
		double sumOfPayoffs = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfPayoffs += discountedPayoffs[j];
		}
		return sumOfPayoffs/numberOfSimulations;
	}
	

	public double getStandardError() {
		double[] discountedPayoffs = getDiscountedPayoffs();
		double meanValue = getMonteCarloValue();
		double sumOfVariance = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfVariance += Math.pow(discountedPayoffs[j] - meanValue, 2.0);
		}
		return Math.sqrt(sumOfVariance/numberOfSimulations)/Math.sqrt(numberOfSimulations);
	}


	public double getSampleVariance() {
		double[] discountedPayoffs = getDiscountedPayoffs();
		double meanValue = getMonteCarloValue();
		double sumOfVariance = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			sumOfVariance += Math.pow(discountedPayoffs[j] - meanValue, 2.0);
		}
		return sumOfVariance/numberOfSimulations;
	}
}