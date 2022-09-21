/**
 * 
 */
package master.thesis.neural.network;

import master.thesis.underlying.UnderlyingPrice;

/**
 * @author QuanLiu
 *
 */
public class GradientDescentAISCall {
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
	
	double learningRate;
	int numberOfIterationTimes;
	
	public GradientDescentAISCall(int numberOfSimulations, int numberOfTimeSteps, int numberOfFirstHiddenLayerNeurons,
			int numberOfSecondHiddenLayerNeurons, double[][] randomNumberMatrix, double initialStockPrice,
			double riskFreeRate, double volatilityTerm, double[] timeSeries, double strike, double maturity,
			double upperBoundFactorB, double upperBoundExponentialDelta1, double lowerBoundFactorA,
			double lowerBoundExponentialDelta2, double learningRate, int numberOfIterationTimes) {
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
		this.learningRate = learningRate;
		this.numberOfIterationTimes = numberOfIterationTimes;
	}
	
	public double[][][][] getOptimalWeightMatrix() {
		NeuralNetworkAISCall neural = new NeuralNetworkAISCall(numberOfSimulations, numberOfTimeSteps, numberOfFirstHiddenLayerNeurons,
				numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
				riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
				upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2);
		UnderlyingPrice normalDistributedRandomNumber = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] normalDistributedRandomNumberMatrix = normalDistributedRandomNumber.getNormalDistributedRandomNumberMatrixUnderTargetDistribution();
		double[][][] weightMatrix1 = neural.initializeWeightMatrix1();
		double[][][] weightMatrix2 = neural.initializeWeightMatrix2();
		double[][][] weightMatrix3 = neural.initializeWeightMatrix3();
		for(int i = 0; i < numberOfIterationTimes; i++) {
			double[][] inputVector = new double[numberOfTimeSteps - 1][timeSeries.length];
			for(int k = 1; k < numberOfTimeSteps - 1; k++) {
				inputVector[k][0] = 1.0;
				for(int j = 1; j < timeSeries.length; j++) {
					inputVector[k][j] = normalDistributedRandomNumberMatrix[j-1][i];
				}
			}
			
			double[][] output1 = neural.getHiddenLayer1Output(inputVector, weightMatrix1);
			double[][] output2 = neural.getHiddenLayer2Output(inputVector, weightMatrix1, weightMatrix2);
			double[][] output3 = neural.getOutput(inputVector, weightMatrix1, weightMatrix2, weightMatrix3);
			double[][][] derivative3 = neural.getDerivativeOfWeightMatrix3(i,weightMatrix1, weightMatrix2, weightMatrix3, output1, output2, output3);
			double[][][] derivative2 = neural.getDerivativeOfWeightMatrix2(i,weightMatrix1, weightMatrix2, weightMatrix3, output1, output2, output3);
			double[][][] derivative1 = neural.getDerivativeOfWeightMatrix1(i, inputVector, weightMatrix1, weightMatrix2, weightMatrix3, output1, output2, output3);
			for(int j = 0; j < weightMatrix3.length; j++) {
				for(int k = 0; k < weightMatrix3[0].length; k++) {
					for(int l = 0; l < weightMatrix3[0][0].length; l++) {
						weightMatrix3[j][k][l] -= learningRate * derivative3[j][k][l];
					}
					
				}
			}
			for(int j = 0; j < weightMatrix2.length; j++) {
				for(int k = 0; k < weightMatrix2[0].length; k++) {
					for(int l = 0; l < weightMatrix2[0][0].length; l++) {
						weightMatrix2[j][k][l] -= learningRate * derivative2[j][k][l];
					}
					
				}
			}
			for(int j = 0; j < weightMatrix1.length; j++) {
				for(int k = 0; k < weightMatrix1[0].length; k++) {
					for(int l = 0; l < weightMatrix1[0][0].length; l++) {
						weightMatrix1[j][k][l] -= learningRate * derivative1[j][k][l];
					}
					
				}
			}
		}
		double[][][][] weightMatrix = {weightMatrix1, weightMatrix2, weightMatrix3};
		return weightMatrix;
	}
	
}
