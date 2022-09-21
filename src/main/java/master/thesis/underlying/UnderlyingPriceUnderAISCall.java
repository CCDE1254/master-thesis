/**
 * 
 */
package master.thesis.underlying;

import master.thesis.neural.network.NeuralNetworkAISCall;
import master.thesis.neural.network.NeuralNetworkISCall;
import net.finmath.functions.NormalDistribution;

/**
 * @author QuanLiu
 *
 */
public class UnderlyingPriceUnderAISCall {
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
	
	double[][][] weightMatrix1;
	double[][][] weightMatrix2;
	double[][][] weightMatrix3;
	

	public UnderlyingPriceUnderAISCall(int numberOfSimulations, int numberOfTimeSteps,
			int numberOfFirstHiddenLayerNeurons, int numberOfSecondHiddenLayerNeurons, double[][] randomNumberMatrix,
			double initialStockPrice, double riskFreeRate, double volatilityTerm, double[] timeSeries, double strike,
			double maturity, double upperBoundFactorB, double upperBoundExponentialDelta1, double lowerBoundFactorA,
			double lowerBoundExponentialDelta2, double[][][] weightMatrix1, double[][][] weightMatrix2,
			double[][][] weightMatrix3) {
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

	public double[][] getIncrementsMatrixUnderAIS(){
		double[][] incrementsMatrixUnderAIS = new double[timeSeries.length-1][numberOfSimulations];
		for(int i = 0; i < numberOfSimulations; i++) {
			NeuralNetworkAISCall neuralNetwork = new NeuralNetworkAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
					numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2);
			double[][] inputVector = new double[numberOfTimeSteps - 1][timeSeries.length];
			for(int k = 1; k < numberOfTimeSteps - 1; k++) {
				inputVector[k][0] = 1.0;
				for(int j = 1; j < timeSeries.length; j++) {
					inputVector[k][j] = NormalDistribution.inverseCumulativeDistribution(randomNumberMatrix[j-1][i]);
				}
			}
			
			double[][] eta = neuralNetwork.getOutput(inputVector, weightMatrix1, weightMatrix2, weightMatrix3);
			for(int j = 0; j < numberOfTimeSteps-1 ; j++) {
				double eta1 = eta[j][0];
				double eta2 = eta[j][1];
				double mean = -2.0*eta1*eta2; //mean
				double variance = -2.0*eta2; //variance
				incrementsMatrixUnderAIS[j][i] = (NormalDistribution.inverseCumulativeDistribution(randomNumberMatrix[j][i])*Math.sqrt(variance) + mean );
			}
		}
		return incrementsMatrixUnderAIS;
	}

	public double[][] getUnderlyingPriceMatrix() {
		double[][] incrementsMatrixUnderAIS = getIncrementsMatrixUnderAIS();
		double[][] underlyingPriceMatrix = new double[timeSeries.length][numberOfSimulations];
		for(int i = 0; i < numberOfSimulations; i++) {
			underlyingPriceMatrix[0][i] = initialStockPrice;
		}
		for(int j = 0; j < numberOfSimulations; j++) {
			for(int i = 1; i < timeSeries.length; i++) {
				underlyingPriceMatrix[i][j] = underlyingPriceMatrix[i-1][j] * Math.exp( (riskFreeRate - (1.0/2)*Math.pow(volatilityTerm, 2))*(timeSeries[i]-timeSeries[i-1]) + 
						volatilityTerm*Math.sqrt(timeSeries[i]-timeSeries[i-1])*(incrementsMatrixUnderAIS[i-1][j]));
			}
		}
		return underlyingPriceMatrix;
	}
}