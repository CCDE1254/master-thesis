/**
 * 
 */
package master.thesis.neural.network;

/**
 * @author QuanLiu
 *
 */
public class GradientDescentForwardAISCall {
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
	
	public GradientDescentForwardAISCall(int numberOfSimulations, int numberOfTimeSteps, int numberOfFirstHiddenLayerNeurons,
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
	
	public double[][] getOptimalNaturalParameterUnderForwardAIS() {
		double[][] optimizedNaturalParameters = new double[numberOfTimeSteps][2];
		for(int j = 0; j < numberOfTimeSteps; j++) {
			optimizedNaturalParameters[j][0] = 0.0;
			optimizedNaturalParameters[j][1] = -1.0/2.0;
		}
		
		for(int c = 1; c < numberOfTimeSteps; c++) {
			NeuralNetworkForwardAISCall neural = new NeuralNetworkForwardAISCall(numberOfSimulations, numberOfTimeSteps, c, optimizedNaturalParameters , numberOfFirstHiddenLayerNeurons,
					numberOfSecondHiddenLayerNeurons, randomNumberMatrix, initialStockPrice,
					riskFreeRate, volatilityTerm, timeSeries, strike, maturity, upperBoundFactorB,
					upperBoundExponentialDelta1, lowerBoundFactorA, lowerBoundExponentialDelta2);
			double[][] weightMatrix1 = neural.initializeWeightMatrix1();
			double[][] weightMatrix2 = neural.initializeWeightMatrix2();
			double[][] weightMatrix3 = neural.initializeWeightMatrix3();
			for(int i = 0; i < numberOfIterationTimes; i++) {
				double[] output1 = neural.getHiddenLayer1Output(weightMatrix1);
				double[] output2 = neural.getHiddenLayer2Output(weightMatrix1, weightMatrix2);
				double[] output3 = neural.getOutput(weightMatrix1, weightMatrix2, weightMatrix3);
				double[][] derivative3 = neural.getDerivativeOfWeightMatrix3(weightMatrix1, weightMatrix2, weightMatrix3, output1, output2, output3);
				double[][] derivative2 = neural.getDerivativeOfWeightMatrix2(weightMatrix1, weightMatrix2, weightMatrix3, output1, output2, output3);
				double[][] derivative1 = neural.getDerivativeOfWeightMatrix1(weightMatrix1, weightMatrix2, weightMatrix3, output1, output2, output3);
				for(int j = 0; j < weightMatrix3.length; j++) {
					for(int k = 0; k < weightMatrix3[0].length; k++) {
						weightMatrix3[j][k] -= learningRate * derivative3[j][k];
					}
				}
				for(int j = 0; j < weightMatrix3.length; j++) {
					for(int k = 0; k < weightMatrix3[0].length; k++) {
						weightMatrix2[j][k] -= learningRate * derivative2[j][k];
					}
				}
				for(int j = 0; j < weightMatrix3.length; j++) {
					for(int k = 0; k < weightMatrix3[0].length; k++) {
						weightMatrix1[j][k] -= learningRate * derivative1[j][k];
					}
				}
			}
			optimizedNaturalParameters[c] = neural.getOutput(weightMatrix1, weightMatrix2, weightMatrix3);
		}
		return optimizedNaturalParameters;
	}
	
}

