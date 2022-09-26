/**
 * 
 */
package master.thesis.underlying;

import net.finmath.functions.NormalDistribution;

/**
 * @author QuanLiu
 *
 */
public class UnderlyingPrice {

	int numberOfSimulations;
	double initialPrice;
	double riskFreeRate;
	double volatility;
	double[] timeSeries;
	double[][] randomNumberMatrix;

	public UnderlyingPrice(int numberOfSimulations, double initialPrice, double riskFreeRate, double volatility,
			double[] timeSeries, double[][] randomNumberMatrix) {
		super();
		this.numberOfSimulations = numberOfSimulations;
		this.initialPrice = initialPrice;
		this.riskFreeRate = riskFreeRate;
		this.volatility = volatility;
		this.timeSeries = timeSeries;
		this.randomNumberMatrix = randomNumberMatrix;
	}

	public double[][] getNormalDistributedRandomNumberMatrixUnderTargetDistribution() {
		double[][] normalDistributedRandomNumberMatrix = new double[timeSeries.length-1][numberOfSimulations];
		for(int j = 0; j < numberOfSimulations; j++) {
			for(int i = 0; i < timeSeries.length-1; i++) {
				normalDistributedRandomNumberMatrix[i][j] = NormalDistribution.inverseCumulativeDistribution(randomNumberMatrix[i][j]);
			}
		}
		return normalDistributedRandomNumberMatrix;
	}
	
	public double[][] getUnderlyingPriceMatrix() {
		double[][] underlyingPriceMatrix = new double[timeSeries.length][numberOfSimulations];
		for(int j = 0; j < numberOfSimulations; j++) {
			underlyingPriceMatrix[0][j] = initialPrice;
		}
		for(int j = 0; j < numberOfSimulations; j++) {
			for(int i = 1; i < timeSeries.length; i++) {
				underlyingPriceMatrix[i][j] = underlyingPriceMatrix[i-1][j] * Math.exp( (riskFreeRate - (1.0/2)*Math.pow(volatility, 2))*(timeSeries[i]-timeSeries[i-1]) + 
						volatility*Math.sqrt(timeSeries[i]-timeSeries[i-1])*NormalDistribution.inverseCumulativeDistribution(randomNumberMatrix[i-1][j]));
			}
		}
		return underlyingPriceMatrix;
	}
}
