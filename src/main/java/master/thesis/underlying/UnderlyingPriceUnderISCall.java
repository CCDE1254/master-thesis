/**
 * 
 */
package master.thesis.underlying;

import net.finmath.functions.NormalDistribution;

/**
 * @author QuanLiu
 *
 */
public class UnderlyingPriceUnderISCall {
	int numberOfSimulations;
	int numberOfTimeSteps;
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
	
	double[] eta;

	public UnderlyingPriceUnderISCall(int numberOfSimulations, int numberOfTimeSteps,
			double[][] randomNumberMatrix,
			double initialStockPrice, double riskFreeRate, double volatilityTerm, double[] timeSeries, double strike,
			double maturity, double upperBoundFactorB, double upperBoundExponentialDelta1, double lowerBoundFactorA,
			double lowerBoundExponentialDelta2, double[] eta) {
		super();
		this.numberOfSimulations = numberOfSimulations;
		this.numberOfTimeSteps = numberOfTimeSteps;
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
		this.eta = eta;
	}

	public double[][] getIncrementsMatrixUnderIS(){
		double[][] incrementsMatrixUnderIS = new double[timeSeries.length-1][numberOfSimulations];
		double eta1 = eta[0];
		double eta2 = eta[1];
		double mean = -eta1/(2.0*eta2); //mean
		double variance = -1.0/(2.0*eta2); //variance
		for(int j = 0; j < numberOfSimulations; j++) {
			for(int i = 0; i < timeSeries.length - 1 ; i++) {
				incrementsMatrixUnderIS[i][j] = (NormalDistribution.inverseCumulativeDistribution(randomNumberMatrix[i][j])*Math.sqrt(variance) + mean );
			}
		}
		return incrementsMatrixUnderIS;
	}
	
	public double[] getNormalParameters(double eta[]){
		double[] normalParameters = new double[2];
			double eta1 = eta[0];
			double eta2 = eta[1];
			double mean = -eta1/(2.0*eta2); //mean
			double variance = -1.0/(2.0*eta2); //variance
			normalParameters[0] = mean;
			normalParameters[1] = variance;
		return normalParameters;
	}

	public double[][] getUnderlyingPriceMatrix() {
		double[][] incrementsMatrixUnderIS = getIncrementsMatrixUnderIS();
		double[][] underlyingPriceMatrix = new double[timeSeries.length][numberOfSimulations];
		for(int i = 0; i < numberOfSimulations; i++) {
			underlyingPriceMatrix[0][i] = initialStockPrice;
		}
		for(int j = 0; j < numberOfSimulations; j++) {
			for(int i = 1; i < timeSeries.length; i++) {
				underlyingPriceMatrix[i][j] = underlyingPriceMatrix[i-1][j] * Math.exp( (riskFreeRate - (1.0/2)*Math.pow(volatilityTerm, 2))*(timeSeries[i]-timeSeries[i-1]) + 
						volatilityTerm*Math.sqrt(timeSeries[i]-timeSeries[i-1])*(incrementsMatrixUnderIS[i-1][j]));
			}
		}
		return underlyingPriceMatrix;
	}
	
	

}
